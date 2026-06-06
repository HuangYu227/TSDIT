"""Offline diagnostics for text-time alignment in MATD CSV datasets.

This module is intentionally analysis-only: it never rewrites captions or
training files.  It answers whether existing text fields carry enough
information to explain coarse time-series statistics before introducing any
LLM-based caption protocol.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np


def _parse_ot(value) -> np.ndarray:
    if isinstance(value, str):
        return np.asarray(ast.literal_eval(value), dtype=np.float32)
    return np.asarray(value, dtype=np.float32)


def _parse_embedding(value) -> np.ndarray:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return np.empty((0,), dtype=np.float32)
    if not isinstance(value, str):
        return np.asarray(value, dtype=np.float32).reshape(-1)
    cleaned = value.strip().strip("[]").replace("\n", " ").replace("\r", " ")
    cleaned = cleaned.replace(",", " ")
    return np.fromstring(cleaned, sep=" ", dtype=np.float32)


def _safe_ratio(num: float, den: float, eps: float = 1e-12) -> float:
    return float(num / max(den, eps))


def _entropy_from_counts(counts: Iterable[int]) -> float:
    values = np.asarray(list(counts), dtype=np.float64)
    total = values.sum()
    if total <= 0:
        return 0.0
    probs = values / total
    probs = probs[probs > 0]
    return float(-(probs * np.log(probs)).sum())


def _normalized_entropy_from_counts(counts: Iterable[int]) -> float:
    values = list(counts)
    if len(values) <= 1:
        return 0.0
    return _safe_ratio(_entropy_from_counts(values), math.log(len(values)))


def _series_stats(x: np.ndarray) -> tuple[np.ndarray, list[str]]:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 2:
        x = x[:, :, None]
    if x.ndim != 3:
        raise ValueError(f"Expected time series shape (N,T) or (N,T,D), got {x.shape}")
    delta = np.diff(x, axis=1)
    centered_t = np.linspace(-0.5, 0.5, x.shape[1], dtype=np.float64)
    t_var = float(np.sum(centered_t * centered_t))
    slope = (x * centered_t[None, :, None]).sum(axis=1) / max(t_var, 1e-12)
    spectrum = np.abs(np.fft.rfft(x - x.mean(axis=1, keepdims=True), axis=1)) ** 2
    if spectrum.shape[1] > 1:
        low_end = max(2, min(4, spectrum.shape[1]))
        low = spectrum[:, 1:low_end].sum(axis=(1, 2))
        total = spectrum[:, 1:].sum(axis=(1, 2))
        low_ratio = low / np.maximum(total, 1e-12)
    else:
        low_ratio = np.zeros((x.shape[0],), dtype=np.float64)

    stats = {
        "mean": x.mean(axis=(1, 2)),
        "std": x.std(axis=(1, 2)),
        "min": x.min(axis=(1, 2)),
        "max": x.max(axis=(1, 2)),
        "range": x.max(axis=(1, 2)) - x.min(axis=(1, 2)),
        "mean_abs_delta": np.abs(delta).mean(axis=(1, 2)) if delta.size else np.zeros(x.shape[0]),
        "max_abs_delta": np.abs(delta).max(axis=(1, 2)) if delta.size else np.zeros(x.shape[0]),
        "trend_slope": slope.mean(axis=1),
        "low_freq_psd_ratio": low_ratio,
    }
    names = list(stats.keys())
    return np.stack([stats[name] for name in names], axis=1), names


def _standardize(x: np.ndarray, eps: float = 1e-8) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.nanmean(x, axis=0, keepdims=True)
    std = np.nanstd(x, axis=0, keepdims=True)
    return (x - mean) / np.maximum(std, eps), mean, std


def _ridge_r2(
    x: np.ndarray,
    y: np.ndarray,
    names: list[str],
    seed: int = 123,
    train_ratio: float = 0.8,
    ridge: float = 1.0,
) -> dict[str, float]:
    finite = np.isfinite(x).all(axis=1) & np.isfinite(y).all(axis=1)
    x = x[finite]
    y = y[finite]
    n = x.shape[0]
    if n < 20 or x.shape[1] == 0:
        return {"text_stat_r2_mean": float("nan"), "text_stat_r2_min": float("nan")}
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    n_train = max(2, min(n - 1, int(n * train_ratio)))
    tr, te = perm[:n_train], perm[n_train:]
    x_train, x_mean, x_std = _standardize(x[tr])
    y_train, y_mean, y_std = _standardize(y[tr])
    x_test = (x[te] - x_mean) / np.maximum(x_std, 1e-8)
    y_test = (y[te] - y_mean) / np.maximum(y_std, 1e-8)
    x_train = np.concatenate([x_train, np.ones((x_train.shape[0], 1))], axis=1)
    x_test = np.concatenate([x_test, np.ones((x_test.shape[0], 1))], axis=1)
    eye = np.eye(x_train.shape[1], dtype=np.float64)
    eye[-1, -1] = 0.0
    weights = np.linalg.pinv(x_train.T @ x_train + ridge * eye) @ x_train.T @ y_train
    pred = x_test @ weights
    ss_res = ((y_test - pred) ** 2).sum(axis=0)
    ss_tot = ((y_test - y_test.mean(axis=0, keepdims=True)) ** 2).sum(axis=0)
    r2 = 1.0 - ss_res / np.maximum(ss_tot, 1e-12)
    out = {
        "text_stat_r2_mean": float(np.mean(r2)),
        "text_stat_r2_min": float(np.min(r2)),
        "text_stat_r2_max": float(np.max(r2)),
    }
    for name, value in zip(names, r2):
        out[f"text_stat_r2/{name}"] = float(value)
    return out


def _kmeans_entropy(x: np.ndarray, k: int = 16, seed: int = 123, iters: int = 25) -> dict[str, float]:
    finite = np.isfinite(x).all(axis=1)
    x = x[finite]
    if x.shape[0] < 2 or x.shape[1] == 0:
        return {"text_embedding_cluster_entropy": float("nan"), "text_embedding_cluster_entropy_norm": float("nan")}
    x, _, _ = _standardize(x)
    k = max(1, min(k, x.shape[0]))
    rng = np.random.RandomState(seed)
    centers = x[rng.choice(x.shape[0], size=k, replace=False)]
    labels = np.zeros(x.shape[0], dtype=np.int64)
    for _ in range(iters):
        dist = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = dist.argmin(axis=1)
        for idx in range(k):
            mask = labels == idx
            if mask.any():
                centers[idx] = x[mask].mean(axis=0)
    counts = np.bincount(labels, minlength=k)
    return {
        "text_embedding_cluster_entropy": _entropy_from_counts(counts),
        "text_embedding_cluster_entropy_norm": _normalized_entropy_from_counts(counts),
        "text_embedding_cluster_count": float(k),
    }


def _embedding_geometry(emb: np.ndarray, sample_pairs: int = 20000, seed: int = 123) -> dict[str, float]:
    finite = np.isfinite(emb).all(axis=1)
    emb = emb[finite]
    if emb.shape[0] < 2 or emb.shape[1] == 0:
        return {
            "text_embedding_dim": float(emb.shape[1] if emb.ndim == 2 else 0),
            "text_embedding_finite_ratio": float(finite.mean()) if finite.size else 0.0,
        }
    centered = emb - emb.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(emb, axis=1)
    rng = np.random.RandomState(seed)
    n_pairs = min(sample_pairs, emb.shape[0] * (emb.shape[0] - 1))
    i = rng.randint(0, emb.shape[0], size=n_pairs)
    j = rng.randint(0, emb.shape[0], size=n_pairs)
    valid = i != j
    cos = (emb[i[valid]] * emb[j[valid]]).sum(axis=1) / np.maximum(norms[i[valid]] * norms[j[valid]], 1e-12)
    return {
        "text_embedding_dim": float(emb.shape[1]),
        "text_embedding_finite_ratio": float(finite.mean()),
        "text_embedding_cov_trace": float(np.var(centered, axis=0).sum()),
        "text_embedding_mean_pair_cos": float(np.mean(cos)) if cos.size else float("nan"),
        "text_embedding_std_pair_cos": float(np.std(cos)) if cos.size else float("nan"),
    }


def _duplicate_text_consistency(texts: list[str], stats: np.ndarray, stat_names: list[str]) -> dict[str, float]:
    groups: dict[str, list[int]] = {}
    for idx, text in enumerate(texts):
        groups.setdefault(text, []).append(idx)
    repeated = [idxs for idxs in groups.values() if len(idxs) > 1]
    if not repeated:
        return {"duplicate_text_group_count": 0.0, "duplicate_text_stat_std_ratio_mean": float("nan")}
    global_std = np.std(stats, axis=0)
    ratios = []
    for idxs in repeated:
        ratios.append(np.std(stats[idxs], axis=0) / np.maximum(global_std, 1e-12))
    ratio = np.mean(np.stack(ratios, axis=0), axis=0)
    out = {
        "duplicate_text_group_count": float(len(repeated)),
        "duplicate_text_sample_ratio": float(sum(len(g) for g in repeated) / max(len(texts), 1)),
        "duplicate_text_stat_std_ratio_mean": float(np.mean(ratio)),
    }
    for name, value in zip(stat_names, ratio):
        out[f"duplicate_text_stat_std_ratio/{name}"] = float(value)
    return out


def load_csv_files(data_dir: str, datasets: list[str], time_interval: int, max_rows: int | None = None):
    import pandas as pd

    frames = []
    for dataset in datasets:
        path = os.path.join(data_dir, f"embedding_cleaned_{dataset}_{time_interval}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        frame["__dataset__"] = dataset
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    if max_rows is not None and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=123).reset_index(drop=True)
    return df


def compute_text_time_diagnostics(df) -> dict[str, float | str]:
    required = {"Text", "OT"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    texts = [str(x) for x in df["Text"].fillna("").tolist()]
    text_counts = Counter(texts)
    series = np.stack([_parse_ot(x) for x in df["OT"].tolist()], axis=0)
    stats, stat_names = _series_stats(series)

    out: dict[str, float | str] = {
        "sample_count": float(len(df)),
        "seq_len": float(series.shape[1]),
        "unique_text_count": float(len(text_counts)),
        "unique_text_ratio": float(len(text_counts) / max(len(texts), 1)),
        "top1_text_fraction": float(max(text_counts.values()) / max(len(texts), 1)),
        "text_entropy": _entropy_from_counts(text_counts.values()),
        "text_entropy_norm": _normalized_entropy_from_counts(text_counts.values()),
        "text_char_len_mean": float(np.mean([len(t) for t in texts])),
        "text_char_len_std": float(np.std([len(t) for t in texts])),
        "ts_mean_abs": float(np.mean(np.abs(series))),
        "ts_std": float(np.std(series)),
        "ts_min": float(np.min(series)),
        "ts_max": float(np.max(series)),
    }
    out.update(_duplicate_text_consistency(texts, stats, stat_names))

    if "TextEmbedding" in df.columns:
        emb_list = [_parse_embedding(x) for x in df["TextEmbedding"].tolist()]
        dims = Counter(int(e.shape[0]) for e in emb_list)
        dim = dims.most_common(1)[0][0] if dims else 0
        valid = [e if e.shape[0] == dim else np.full((dim,), np.nan, dtype=np.float32) for e in emb_list]
        emb = np.stack(valid, axis=0).astype(np.float64) if dim > 0 else np.empty((len(df), 0))
        out.update(_embedding_geometry(emb))
        out.update(_kmeans_entropy(emb))
        out.update(_ridge_r2(emb, stats, stat_names))
    else:
        out["text_embedding_dim"] = 0.0
        out["text_stat_r2_mean"] = float("nan")
    return out


def _format_markdown(results: dict[str, float | str]) -> str:
    lines = ["# Text-Time Alignment Diagnostics", ""]
    key_order = [
        "sample_count",
        "seq_len",
        "unique_text_count",
        "unique_text_ratio",
        "top1_text_fraction",
        "text_entropy_norm",
        "text_embedding_dim",
        "text_embedding_cov_trace",
        "text_embedding_mean_pair_cos",
        "text_embedding_cluster_entropy_norm",
        "text_stat_r2_mean",
        "duplicate_text_stat_std_ratio_mean",
        "ts_mean_abs",
        "ts_std",
        "ts_min",
        "ts_max",
    ]
    for key in key_order:
        if key in results:
            value = results[key]
            if isinstance(value, float):
                lines.append(f"- `{key}`: {value:.6g}")
            else:
                lines.append(f"- `{key}`: {value}")
    lines.append("")
    lines.append("## Per-Statistic Predictability")
    for key in sorted(k for k in results if k.startswith("text_stat_r2/")):
        value = results[key]
        lines.append(f"- `{key}`: {float(value):.6g}")
    lines.append("")
    lines.append("## Duplicate-Text Consistency")
    for key in sorted(k for k in results if k.startswith("duplicate_text_stat_std_ratio/")):
        value = results[key]
        lines.append(f"- `{key}`: {float(value):.6g}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose text-time alignment for MATD CSV datasets.")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--datasets", required=True, help="Comma-separated dataset names, e.g. airquality,traffic")
    parser.add_argument("--time_interval", type=int, required=True)
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--output", type=str, default=None, help="Optional JSON output path.")
    parser.add_argument("--markdown_output", type=str, default=None, help="Optional Markdown summary path.")
    args = parser.parse_args()

    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    df = load_csv_files(args.data_dir, datasets, args.time_interval, max_rows=args.max_rows)
    results = compute_text_time_diagnostics(df)
    results["datasets"] = ",".join(datasets)
    results["data_dir"] = args.data_dir

    text = json.dumps(results, ensure_ascii=False, indent=2, allow_nan=True)
    print(text)
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")
    if args.markdown_output:
        md_path = Path(args.markdown_output)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(_format_markdown(results), encoding="utf-8")


if __name__ == "__main__":
    main()
