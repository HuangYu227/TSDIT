"""Synthetic benchmark for the dynamic CTICD graph learner.

This experiment validates whether the lagged graph learner can recover a known
mechanism-level dynamic graph from observational temporal states.  It is meant
as a module-level causal sanity check for the paper, complementing downstream
TIGER generation metrics.

Example:
    python -m TIGER_paper_ready.experiments.causal_synthetic_benchmark \
        --steps 1500 --nodes 12 --dim 16 --max_lag 2 --batch_size 64
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from ..cticd import DynamicCausalGraphLearner, LaggedMechanismPredictor


def make_ground_truth(nodes: int, max_lag: int, edge_prob: float, seed: int, device) -> torch.Tensor:
    g = torch.Generator(device=device).manual_seed(seed)
    gt = (torch.rand(max_lag, nodes, nodes, generator=g, device=device) < edge_prob).float()
    # self-lag is allowed but downweighted to avoid trivial identity-only dynamics
    eye = torch.eye(nodes, device=device).unsqueeze(0)
    gt = torch.where(eye.bool(), gt * 0.5, gt)
    return gt


def simulate_states(
    n_series: int,
    length: int,
    nodes: int,
    dim: int,
    max_lag: int,
    gt: torch.Tensor,
    noise_std: float,
    seed: int,
    device,
) -> torch.Tensor:
    g = torch.Generator(device=device).manual_seed(seed + 13)
    weights = torch.randn(max_lag, nodes, nodes, dim, generator=g, device=device) * gt[..., None] * 0.35
    x = torch.randn(n_series, length, nodes, dim, generator=g, device=device) * 0.1
    for t in range(max_lag, length):
        val = 0.15 * x[:, t - 1]
        for lag in range(1, max_lag + 1):
            parent = torch.tanh(x[:, t - lag])  # (N,M,D)
            # gt convention: parent j -> child i
            val = val + torch.einsum("njd,jid->nid", parent, weights[lag - 1])
        val = val + noise_std * torch.randn(n_series, nodes, dim, generator=g, device=device)
        x[:, t] = val.clamp(-5, 5)
    return x


def sample_windows(states: torch.Tensor, window: int, batch_size: int) -> torch.Tensor:
    N, T, M, D = states.shape
    series_idx = torch.randint(0, N, (batch_size,), device=states.device)
    start_idx = torch.randint(0, T - window + 1, (batch_size,), device=states.device)
    batch = []
    for n, s in zip(series_idx.tolist(), start_idx.tolist()):
        batch.append(states[n, s:s + window])
    return torch.stack(batch, dim=0)


def graph_metrics(pred: torch.Tensor, gt: torch.Tensor, threshold: float) -> dict:
    pred_bin = (pred >= threshold).float()
    gt_bin = (gt > 0).float()
    tp = (pred_bin * gt_bin).sum().item()
    fp = (pred_bin * (1 - gt_bin)).sum().item()
    fn = ((1 - pred_bin) * gt_bin).sum().item()
    tn = ((1 - pred_bin) * (1 - gt_bin)).sum().item()
    precision = tp / max(tp + fp, 1e-8)
    recall = tp / max(tp + fn, 1e-8)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    shd = fp + fn
    return {"precision": precision, "recall": recall, "f1": f1, "shd": shd, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--nodes", type=int, default=12)
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--max_lag", type=int, default=2)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_series", type=int, default=512)
    parser.add_argument("--length", type=int, default=32)
    parser.add_argument("--edge_prob", type=float, default=0.12)
    parser.add_argument("--noise_std", type=float, default=0.03)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="./runs/cticd_synthetic/results.json")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gt = make_ground_truth(args.nodes, args.max_lag, args.edge_prob, args.seed, device)
    states = simulate_states(args.n_series, args.length, args.nodes, args.dim, args.max_lag, gt, args.noise_std, args.seed, device)

    learner = DynamicCausalGraphLearner(args.nodes, args.dim, max_lag=args.max_lag).to(device)
    predictor = LaggedMechanismPredictor(args.dim, args.nodes, attr_dim=args.dim, max_lag=args.max_lag).to(device)
    opt = torch.optim.AdamW(list(learner.parameters()) + list(predictor.parameters()), lr=args.lr, weight_decay=1e-4)

    attr = torch.zeros(args.batch_size, args.dim, device=device)
    for step in range(1, args.steps + 1):
        batch = sample_windows(states, args.window, args.batch_size)
        A0, Alags, notears, sparsity = learner(batch)
        pred, target, _ = predictor(batch, Alags, attr)
        loss_pred = F.mse_loss(pred, target.detach())
        loss = loss_pred + 1e-3 * notears + 1e-2 * sparsity
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(learner.parameters()) + list(predictor.parameters()), 1.0)
        opt.step()
        if step % 200 == 0 or step == 1:
            print(f"step={step:05d} loss={loss.item():.5f} pred={loss_pred.item():.5f} notears={notears.item():.5f}")

    with torch.no_grad():
        eval_batch = sample_windows(states, args.window, min(256, args.n_series))
        _, Alags, _, _ = learner(eval_batch)
        pred_graph = Alags.mean(dim=0)
        metrics = graph_metrics(pred_graph, gt, args.threshold)
        metrics.update({
            "threshold": args.threshold,
            "edge_density_pred": float(pred_graph.mean().item()),
            "edge_density_gt": float((gt > 0).float().mean().item()),
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
