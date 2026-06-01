"""TIGER Training Pipeline.

Trains the text-conditioned diffusion model for time-series generation.

Usage:
    python -m mmldm.tiger.train --data_dir <path> --config <yaml_or_json>

Or with argparse defaults (Weather dataset):
    python -m mmldm.tiger.train \
        --data_dir "./Three Levels Data/Weather" \
        --dataset_type weather_npy \
        --epochs 500 --batch_size 64 --lr 3e-4
"""

import os
import time
import json
import argparse
import random

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .data.dataset import TIGERDataset, TIGERCollateFn
from .generator import TIGERGenerator
from .image_to_ts import ImageToTSDecoder
from .ts_to_image import NormParams


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def get_default_config() -> dict:
    """Return a complete default config dict (Weather dataset)."""
    return {
        "device": "cuda:0",
        "seed": 42,
        "epochs": 500,
        "batch_size": 64,
        "lr": 3e-4,
        "weight_decay": 1e-6,
        "warmup_steps": 500,
        "val_interval": 10,
        "display_interval": 10,
        "save_interval": 50,
        "log_dir": "./runs/tiger",
        "save_dir": "./checkpoints/tiger",
        "model_path": "",
        "eval_only": False,

        "diffusion": {
            "num_steps": 50,
            "beta_start": 0.0001,
            "beta_end": 0.02,
            "schedule": "quad",
            "channels": 256,
            "nheads": 8,
            "layers": 12,
            "n_var": 1,
            "multipatch_num": 1,
            "base_patch": 4,
            "patch_scale": 2,
            "patch_mode": "row_raster",
            "ff_mult": 4,
            "diffusion_embedding_dim": 512,
            "in_channels": 1,
            "condition_type": "adaLN",
            "attention_mask_type": "parallel",
            "lambda_x0_ts": 0.2,
            "x0_delta_weight": 0.5,
            "lambda_cticd": 0.1,
            "lambda_moe": 0.05,
            "cticd_enable_threshold": 0.5,
        },

        "condition": {
            # Paper-ready default: true text+image conditioning.
            # Use "text_only" only for baseline ablations.
            "cond_mode": "text_image",
            "num_stages": 4,
            "cfg_dropout": 0.10,
            "drop_text_prob": 0.10,
            "drop_image_prob": 0.10,
            "drop_both_prob": 0.10,
            "joint_emb": 256,
            "fusion_heads": 8,
            "fusion_layers": 2,
            "reference": {
                "mode": "masked_self",
                "mask_ratio": 0.50,
                "noise_std": 0.02,
                "fill": 0.5,
            },
            "decode_scale_mode": "global",  # avoids oracle per-sample min/max at test time
            "text_guidance_scale": 1.0,
            "image_guidance_scale": 1.0,
            "interaction_guidance_scale": 0.0,
            "text": {
                "pretrain_model_path": "openai/clip-vit-base-patch32",
                "pretrain_model_dim": 512,
                "textemb_hidden_dim": 512,
                "text_emb": 256,
            },
            "image": {
                "encoder": "vit",
                "img_size": 64,
                "patch_size": 8,
                "embed_dim": 384,
                "depth": 6,
                "num_heads": 8,
                "image_emb": 256,
            },
        },

        "csa_moe": {
            "enabled": True,
            "k": 1,
            "alpha": 0.01,
            "inject_aux": False,
            "scca_heads": 8,
        },

        "cticd": {
            "enabled": True,
            "d_model": 256,
            "n_channels": 1,
            "n_mechanisms_per_channel": 6,
            "n_segments": 8,
            "max_lag": 2,
            "patch_size": 2,
            "num_heads": 4,
            "edge_bias": -4.0,
            "lag_edge_bias": -2.5,
            "branch_grad_scale": 0.5,
            "lambda_causal": 1.0,
            "lambda_notears": 0.05,
            "lambda_sparsity": 1e-2,
            "lambda_smooth": 1e-3,
            "lambda_prior": 0.1,
            "use_lag_prior": True,
            "lag_prior_method": "acf",
            "lag_prior_mode": "bias_gate",
            "lag_prior_strength": 1.0,
            "lag_prior_significance": True,
            "lag_prior_bins": 16,
            "lag_topk": 6,
            "injection_init": -2.0,
        },

        "data": {
            "dataset_type": "weather_npy",
            "representation": "row_raster",
            "patch_size": 8,          # legacy fallback for old configs
            "layout_patch_size": 2,   # row-raster width alignment
            "decode_mode": "row_raster",
            "image_size": 64,
            "n_fft": 64,
            "hop_length": 8,
            "epsilon_quantile": 0.1,
        },
    }


def load_config(path: str) -> dict:
    """Load config from JSON file and merge with defaults."""
    defaults = get_default_config()
    if not path or not os.path.exists(path):
        return defaults
    with open(path, "r") as f:
        user_cfg = json.load(f)

    def _merge(base: dict, override: dict):
        for k, v in override.items():
            if k in base and isinstance(base[k], dict) and isinstance(v, dict):
                _merge(base[k], v)
            else:
                base[k] = v

    _merge(defaults, user_cfg)
    return defaults


def _set_if_not_none(config: dict, key: str, value):
    if value is not None:
        config[key] = value


def _update_if_not_none(config: dict, values: dict):
    for key, value in values.items():
        if value is not None:
            config[key] = value


def apply_cli_overrides(config: dict, args: argparse.Namespace) -> dict:
    """Apply only explicitly supplied CLI values on top of JSON/default config."""
    config["data_dir"] = args.data_dir

    _update_if_not_none(config, {
        "save_dir": args.save_dir,
        "log_dir": args.log_dir,
        "model_path": args.model_path,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "warmup_steps": args.warmup_steps,
        "seed": args.seed,
        "val_interval": args.val_interval,
        "display_interval": args.display_interval,
        "save_interval": args.save_interval,
        "eval_only": args.eval_only,
    })
    _update_if_not_none(config["data"], {
        "dataset_type": args.dataset_type,
        "datasets": args.datasets,
        "time_interval": args.time_interval,
        "image_size": args.image_size,
        "n_fft": args.n_fft,
        "hop_length": args.hop_length,
    })
    _update_if_not_none(config["diffusion"], {
        "num_steps": args.num_steps,
        "channels": args.channels,
        "nheads": args.nheads,
        "layers": args.layers,
        "n_var": args.n_var,
        "multipatch_num": args.multipatch_num,
    })
    return config


def _squeeze_trailing_singletons(x: torch.Tensor) -> torch.Tensor:
    """Collapse accidental (..., 1) norm tensors to prevent batch broadcasting."""
    while x.dim() > 1 and x.shape[-1] == 1:
        x = x.squeeze(-1)
    return x


def denormalize_ts_batch(
    ts_norm: torch.Tensor,
    ts_min: torch.Tensor,
    ts_max: torch.Tensor,
) -> torch.Tensor:
    """Denormalize (B, T) time series with scalar or per-variate min/max."""
    ts_min = _squeeze_trailing_singletons(ts_min)
    ts_max = _squeeze_trailing_singletons(ts_max)

    if ts_min.dim() == 1:
        ts_min = ts_min.unsqueeze(-1)
        ts_max = ts_max.unsqueeze(-1)
    elif ts_min.dim() != 2:
        raise ValueError(f"Expected ts_min/ts_max to be 1D or 2D, got {ts_min.shape}")

    return ts_norm * (ts_max - ts_min) + ts_min


def _scalar_float(value):
    """Return a Python float for scalar tensor/numpy/number values."""
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        return float(value.detach().cpu().item())
    if isinstance(value, np.ndarray):
        if value.size != 1:
            return None
        return float(value.reshape(-1)[0])
    if isinstance(value, np.generic):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _json_safe(value):
    """Convert tensors/numpy scalars/arrays in metric dicts to JSON-safe values."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        return value.item() if value.numel() == 1 else value.tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    return value


def _ensure_btd(array: np.ndarray) -> np.ndarray:
    """Normalize saved/evaluated time-series arrays to (B, T, D)."""
    if array.ndim == 2:
        return array[:, :, None]
    if array.ndim == 3:
        return array
    raise ValueError(f"Expected array with shape (B,T) or (B,T,D), got {array.shape}")


# ---------------------------------------------------------------------------
# Learning rate schedule with linear warmup + cosine decay
# ---------------------------------------------------------------------------

class WarmupCosineScheduler:
    """Linear warmup then cosine decay to 0."""

    def __init__(self, optimizer, warmup_steps: int, total_steps: int):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self.step_count = 0

    def step(self):
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            scale = self.step_count / max(1, self.warmup_steps)
        else:
            progress = (self.step_count - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            scale = 0.5 * (1.0 + np.cos(np.pi * progress))
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = base_lr * scale

    def get_lr(self):
        return self.optimizer.param_groups[0]["lr"]

    def state_dict(self) -> dict:
        return {"step_count": self.step_count, "base_lrs": self.base_lrs}

    def load_state_dict(self, d: dict):
        self.step_count = d["step_count"]
        self.base_lrs = d["base_lrs"]


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def calc_mse(real: np.ndarray, gen: np.ndarray) -> float:
    """MSE between real and generated time series. Both (B, T)."""
    return float(np.mean((real - gen) ** 2))


def calc_wape(real: np.ndarray, gen: np.ndarray) -> float:
    """WAPE macro: per-sample WAPE then nanmean. Matches T2S calculate_wape."""
    real = real.reshape(real.shape[0], -1)
    gen = gen.reshape(gen.shape[0], -1)
    values = []
    for i in range(real.shape[0]):
        denom = np.sum(np.abs(real[i]))
        if denom != 0:
            values.append(np.sum(np.abs(real[i] - gen[i])) / denom)
        else:
            values.append(np.nan)
    return float(np.nanmean(values))


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class TIGERTrainer:
    """End-to-end trainer for TIGERGenerator."""

    def __init__(self, config: dict):
        self.config = config
        self.device = config["device"]
        self.n_epochs = config["epochs"]
        self.val_interval = config["val_interval"]
        self.display_interval = config["display_interval"]
        self.save_interval = config["save_interval"]
        self.eval_gen_interval = config.get("eval_gen_interval", config.get("val_interval", 10))

        if not config.get("eval_only", False):
            os.makedirs(config["save_dir"], exist_ok=True)
        os.makedirs(config["log_dir"], exist_ok=True)

        self._init_data()
        self._init_model()
        if config.get("eval_only", False):
            self.optimizer = None
            self.scheduler = None
        else:
            self._init_opt()
        self._init_logging()

        dc = self.config["data"]
        representation = dc.get("representation", "gasf")
        if representation == "row_raster":
            from .image_to_ts import RowRasterDecoder
            self.decoder = RowRasterDecoder().to(self.device)
        else:
            decode_mode = dc.get("decode_mode", "gasf")
            if decode_mode not in ("gasf", "fused"):
                raise ValueError(f"decode_mode must be 'gasf' or 'fused', got '{decode_mode}'")
            self.decoder = ImageToTSDecoder(
                mode=decode_mode, n_fft=dc["n_fft"], hop_length=dc["hop_length"]
            ).to(self.device)

        self.best_val_loss = float("inf")
        self.global_step = 0

        # Metrics history (saved to JSON)
        self.metrics_history: list = []
        self.metrics_path = os.path.join(config.get("log_dir", "."), "metrics.json")

    # ---- data ---------------------------------------------------------------

    def _row_raster_layout_patch_size(self) -> int:
        dc = self.config["data"]
        if dc.get("representation", "gasf") != "row_raster":
            return int(dc.get("patch_size", 8))
        return int(
            dc.get(
                "layout_patch_size",
                self.config.get("diffusion", {}).get("base_patch", dc.get("patch_size", 2)),
            )
        )

    def _init_data(self):
        dc = self.config["data"]
        common = dict(
            data_dir=self.config["data_dir"],
            dataset_type=dc["dataset_type"],
            image_size=dc["image_size"],
            n_fft=dc["n_fft"],
            hop_length=dc["hop_length"],
            epsilon_quantile=dc["epsilon_quantile"],
            datasets=dc.get("datasets"),
            time_interval=dc.get("time_interval", 24),
            representation=dc.get("representation", "gasf"),
            patch_size=self._row_raster_layout_patch_size() if dc.get("representation", "gasf") == "row_raster" else dc.get("patch_size", 8),
        )
        # ── split naming ──────────────────────────────────────────────────
        # Weather   .npy files: train / valid / test
        # CSV       with 'split' column: train / val / test
        # CSV       without 'split' column: random three-way split via split_ratio
        val_name = "valid" if dc["dataset_type"] == "weather_npy" else "val"
        # ────────────────────────────────────────────────────────────────────

        collate = TIGERCollateFn()

        # --- train split ---
        if self.config.get("eval_only", False):
            self.train_loader = None
            self.train_dataset = None
        else:
            train_ds = TIGERDataset(split="train", **common)
            self.train_loader = DataLoader(
                train_ds, batch_size=self.config["batch_size"],
                shuffle=True, collate_fn=collate, num_workers=0,
                drop_last=True,
            )
            self.train_dataset = train_ds

        # --- validation split (for early stopping / hyperparameter selection) ---
        val_ds = TIGERDataset(split=val_name, **common)
        self.val_loader = DataLoader(
            val_ds, batch_size=self.config["batch_size"],
            shuffle=False, collate_fn=collate, num_workers=0,
        )

        # --- test split (held out; only used by evaluate_only / evaluator.py) ---
        test_ds = TIGERDataset(split="test", **common)
        self.test_loader = DataLoader(
            test_ds, batch_size=self.config["batch_size"],
            shuffle=False, collate_fn=collate, num_workers=0,
        )

    # ---- model --------------------------------------------------------------

    def _init_model(self):
        model_config = {
            "device": self.device,
            "diffusion": dict(self.config["diffusion"]),
            "condition": self.config["condition"],
        }
        # Pass top-level csa_moe / cticd configs so dit_model can read them
        if "csa_moe" in self.config:
            model_config["diffusion"]["csa_moe"] = self.config["csa_moe"]
        if "cticd" in self.config:
            model_config["diffusion"]["cticd"] = dict(self.config["cticd"])
        # Pass image_size for MoE grid computation
        if "data" in self.config and "image_size" in self.config["data"]:
            model_config["diffusion"]["image_size"] = self.config["data"]["image_size"]
        # For row-raster: pass separate H/W
        if "data" in self.config and self.config["data"].get("representation") == "row_raster":
            from .ts_to_image import RowRasterEncoder
            T = self.config["data"].get("time_interval", 24)
            H, W = RowRasterEncoder._optimal_hw(T, self._row_raster_layout_patch_size())
            model_config["diffusion"]["image_size_h"] = H
            model_config["diffusion"]["image_size_w"] = W
            model_config["diffusion"]["signal_length"] = int(T)
            model_config["diffusion"]["patch_mode"] = "row_raster"
            base_patch = int(model_config["diffusion"].get("base_patch", self._row_raster_layout_patch_size()))
            base_patch = max(1, min(base_patch, int(W)))
            model_config["diffusion"]["base_patch"] = base_patch
            # In row-raster mode the "variable" anchor axis is reused as row
            # anchors and condition_n_scale is reused as horizontal temporal
            # patch anchors, so image/text conditions align with the actual
            # row-raster token grid instead of broadcasting over columns.
            model_config["diffusion"]["n_var"] = int(H)
            model_config["diffusion"]["condition_n_scale"] = int((W + base_patch - 1) // base_patch)
            if not model_config["diffusion"].get("allow_multipatch_raster", False):
                model_config["diffusion"]["multipatch_num"] = 1
            if "cticd" in model_config["diffusion"]:
                model_config["diffusion"]["cticd"]["signal_length"] = int(T)
                cticd_patch = int(model_config["diffusion"]["cticd"].get("patch_size", base_patch))
                model_config["diffusion"]["cticd"]["patch_size"] = max(1, min(cticd_patch, int(W)))
        self.model = TIGERGenerator(model_config)

        if self.config.get("model_path"):
            print(f"Loading checkpoint from {self.config['model_path']}")
            ckpt = torch.load(self.config["model_path"], map_location=self.device)
            # Support both old (state_dict only) and new (dict) checkpoint formats
            if isinstance(ckpt, dict) and "model" in ckpt:
                missing, unexpected = self.model.load_state_dict(ckpt["model"], strict=False)
                if missing:
                    print(f"WARNING: Checkpoint missing keys ({len(missing)}): {missing[:5]}...")
                if unexpected:
                    print(f"WARNING: Checkpoint unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
                # Resume optimizer/scheduler state for continued training
                if not self.config.get("eval_only", False) and self.optimizer is not None:
                    if ckpt.get("optimizer") is not None:
                        self.optimizer.load_state_dict(ckpt["optimizer"])
                        print("  Restored optimizer state")
                    if ckpt.get("scheduler") is not None and self.scheduler is not None:
                        self.scheduler.load_state_dict(ckpt["scheduler"])
                        print("  Restored scheduler state")
                    if ckpt.get("global_step") is not None:
                        self.global_step = ckpt["global_step"]
                        print(f"  Restored global_step={self.global_step}")
                    if ckpt.get("val_loss") is not None:
                        self.best_val_loss = ckpt["val_loss"]
                        print(f"  Restored best_val_loss={self.best_val_loss:.6f}")
            else:
                missing, unexpected = self.model.load_state_dict(ckpt, strict=False)
                if missing:
                    print(f"WARNING: Checkpoint missing keys ({len(missing)}): {missing[:5]}...")
                if unexpected:
                    print(f"WARNING: Checkpoint unexpected keys ({len(unexpected)}): {unexpected[:5]}...")

    # ---- optimizer ----------------------------------------------------------

    def _init_opt(self):
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = Adam(
            trainable_params,
            lr=self.config["lr"],
            weight_decay=self.config["weight_decay"],
        )
        steps_per_epoch = len(self.train_loader)
        total_steps = self.n_epochs * steps_per_epoch
        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            warmup_steps=self.config.get("warmup_steps", 500),
            total_steps=total_steps,
        )

    # ---- logging ------------------------------------------------------------

    def _init_logging(self):
        self.writer = SummaryWriter(log_dir=self.config["log_dir"])

    # ---- train loop ---------------------------------------------------------

    def train(self):
        if self.config.get("eval_only", False):
            raise RuntimeError("train() called in eval_only mode; use evaluate_only().")

        print(f"Starting training for {self.n_epochs} epochs")
        print(f"  Train samples: {len(self.train_loader.dataset)}")
        print(f"  Val samples:   {len(self.val_loader.dataset)}")
        print(f"  Batch size:    {self.config['batch_size']}")
        print(f"  Device:        {self.device}")
        print(f"  Early stop lr: {self.config.get('early_stop_lr', 1e-6):.2e}")

        for epoch in range(self.n_epochs):
            train_loss = self._train_epoch(epoch)

            # Early stopping signal
            if not np.isfinite(train_loss):
                print(f"\n{'='*60}")
                print(f"TRAINING STOPPED EARLY at epoch {epoch + 1}")
                print(f"Reason: Loss underflow/NaN at low learning rate")
                print(f"{'='*60}")
                break

            epoch_metrics = {"epoch": epoch + 1, "train_loss": train_loss}
            epoch_metrics.update(getattr(self, "_last_train_epoch_metrics", {}))

            # Append BEFORE _validate so gen_metrics merge targets the current epoch.
            self.metrics_history.append(epoch_metrics)

            if (epoch + 1) % self.val_interval == 0:
                val_loss = self._validate(epoch)
                epoch_metrics["val_loss"] = val_loss
                epoch_metrics.update(getattr(self, "_last_val_epoch_metrics", {}))
                self._save_checkpoint(epoch, val_loss)

            if (epoch + 1) % self.save_interval == 0:
                self._save_checkpoint(epoch, tag=f"epoch_{epoch+1}")

            with open(self.metrics_path, "w") as f:
                json.dump(_json_safe(self.metrics_history), f, indent=2)

        self.writer.close()
        print("Training complete.")

    def evaluate_only(self):
        """Evaluate a checkpoint on the HELD-OUT TEST SET.

        The test set is never used during training or hyperparameter selection
        (which use val_loader).  This method is for final checkpoint evaluation
        only — it reports metrics on data the model has never seen.
        """
        if not self.config.get("model_path"):
            raise ValueError("--eval_only requires --model_path")

        print("Starting eval-only run on TEST SET (held-out data)")
        print(f"  Test samples: {len(self.test_loader.dataset)}")
        print(f"  Batch size:   {self.config['batch_size']}")
        print(f"  Device:       {self.device}")
        print(f"  Checkpoint:   {self.config['model_path']}")

        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        loss_sums = {}

        for batch in tqdm(self.test_loader, desc="Evaluating on test"):
            loss_dict = self.model(batch, is_train=False)
            if not torch.isfinite(loss_dict["all"]):
                print("WARNING: NaN/INF test loss, skipping batch")
                continue
            total_loss += loss_dict["all"].item()
            num_batches += 1
            for k, v in loss_dict.items():
                scalar = _scalar_float(v)
                if scalar is not None and np.isfinite(scalar):
                    loss_sums[k] = loss_sums.get(k, 0.0) + scalar

        avg_loss = total_loss / num_batches if num_batches else float("inf")
        self.writer.add_scalar("test/loss", avg_loss, 0)
        loss_avgs = {k: v / max(1, num_batches) for k, v in loss_sums.items()}
        for k, v in loss_avgs.items():
            self.writer.add_scalar(f"test_epoch/{k}", v, 0)
        print(f"Test loss = {avg_loss:.6f}")

        # Compute full gen metrics on test set
        gen_metrics = self._compute_gen_metrics(
            epoch=0,
            loader=self.test_loader,
            prefix="test",
        )
        for k, v in gen_metrics.items():
            scalar = _scalar_float(v)
            if scalar is not None and np.isfinite(scalar):
                self.writer.add_scalar(f"test/{k}", scalar, 0)

        self.writer.close()
        print(f"Eval complete. test_loss={avg_loss:.6f}")

        # Save results
        results = {
            "test_loss": avg_loss,
            **{f"test_{k}": v for k, v in loss_avgs.items() if k != "all"},
            **gen_metrics,
        }
        results_path = os.path.join(
            self.config.get("log_dir", "."), "test_results.json",
        )
        with open(results_path, "w") as f:
            json.dump(_json_safe(results), f, indent=2)
        print(f"Test results saved to {results_path}")
        return results

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        self._last_train_epoch_metrics = {}
        total_loss = 0.0
        num_updates = 0
        num_skipped = 0
        t0 = time.time()
        loss_sums = {}

        # Early stopping config
        early_stop_lr = self.config.get("early_stop_lr", 1e-6)
        max_skip_ratio = self.config.get("max_skip_ratio", 0.5)

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            self.optimizer.zero_grad()
            loss_dict = self.model(batch, is_train=True)
            loss = loss_dict["all"]
            # Skip NaN/Inf (caught by isfinite) and genuine underflow.
            # Threshold of 1e-15 is safely below any legitimate float32 loss:
            # float32 machine epsilon is ~1.19e-7, but loss values near 0 are
            # well-represented in float32 (smallest normalized ~1.17e-38).
            # 1e-15 catches values approaching the subnormal range while
            # allowing well-converged models to keep learning at low LR.
            # The NaN/Inf check and the gradient-finiteness check below already
            # catch the real problems.
            if not torch.isfinite(loss) or loss.item() < 1e-15:
                reason = "NaN/INF" if not torch.isfinite(loss) else "underflow"
                num_skipped += 1
                current_lr = self.scheduler.get_lr()
                if current_lr <= early_stop_lr:
                    print(f"\nEARLY STOP: lr={current_lr:.2e} <= {early_stop_lr:.2e} and {reason} loss at step {self.global_step}")
                    print(f"  Skipped {num_skipped}/{num_skipped + num_updates} batches this epoch")
                    return float("inf")  # Signal to stop training
                print(f"WARNING: {reason} loss={loss.item():.2e} at step {self.global_step}, skipping batch")
                self.optimizer.zero_grad()
                continue
            loss.backward()

            grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            if not torch.isfinite(grad_norm):
                num_skipped += 1
                current_lr = self.scheduler.get_lr()
                if current_lr <= early_stop_lr:
                    print(f"\nEARLY STOP: lr={current_lr:.2e} <= {early_stop_lr:.2e} and NaN/INF grad at step {self.global_step}")
                    print(f"  Skipped {num_skipped}/{num_skipped + num_updates} batches this epoch")
                    return float("inf")  # Signal to stop training
                print(f"WARNING: NaN/INF grad at step {self.global_step}, skipping optimizer step")
                self.optimizer.zero_grad()
                continue

            self.optimizer.step()
            self.scheduler.step()
            self.global_step += 1

            total_loss += loss_dict["all"].item()
            num_updates += 1
            for k, v in loss_dict.items():
                scalar = _scalar_float(v)
                if scalar is not None and np.isfinite(scalar):
                    loss_sums[k] = loss_sums.get(k, 0.0) + scalar

            # Build compact postfix: total loss + key sub-losses if present
            postfix = {"loss": f"{loss_dict['all'].item():.4f}"}
            if "noise_loss" in loss_dict:
                postfix["noise"] = f"{loss_dict['noise_loss'].item():.4f}"
            if "x0_weighted" in loss_dict:
                postfix["x0"] = f"{loss_dict['x0_weighted'].item():.4f}"
            if "cticd_weighted" in loss_dict:
                postfix["cticd"] = f"{loss_dict['cticd_weighted'].item():.4f}"
            if "moe_aux" in loss_dict:
                postfix["moe"] = f"{loss_dict['moe_aux'].item():.4f}"
            # Additional CTICD diagnostic fields
            if "cticd_total" in loss_dict:
                postfix["cticd_t"] = f"{loss_dict['cticd_total'].item():.4f}"
            if "cticd_prior" in loss_dict:
                postfix["prior"] = f"{loss_dict['cticd_prior'].item():.4f}"
            if "cticd_lag_edge_density" in loss_dict:
                postfix["lag_e"] = f"{loss_dict['cticd_lag_edge_density'].item():.3f}"
            if "cticd_lag_prior_mean" in loss_dict:
                postfix["lag_p"] = f"{loss_dict['cticd_lag_prior_mean'].item():.3f}"
            if "cticd_lag_active_density" in loss_dict:
                postfix["lag_a"] = f"{loss_dict['cticd_lag_active_density'].item():.3f}"
            if "cticd_feature_ratio" in loss_dict:
                postfix["c_fr"] = f"{loss_dict['cticd_feature_ratio'].item():.3f}"
            postfix["grad"] = f"{grad_norm:.2f}"
            postfix["lr"] = f"{self.scheduler.get_lr():.2e}"
            pbar.set_postfix(postfix)

            # Log per-step
            for k, v in loss_dict.items():
                scalar = _scalar_float(v)
                if scalar is not None and np.isfinite(scalar):
                    self.writer.add_scalar(f"train_step/{k}", scalar, self.global_step)
            self.writer.add_scalar("train_step/lr", self.scheduler.get_lr(), self.global_step)

        # Check skip ratio at end of epoch
        total_batches = num_updates + num_skipped
        skip_ratio = num_skipped / total_batches if total_batches > 0 else 0
        if skip_ratio > max_skip_ratio:
            print(f"\nWARNING: High skip ratio {skip_ratio:.1%} ({num_skipped}/{total_batches}) at epoch {epoch+1}")

        # If no batches produced valid gradients, stop early
        if num_updates == 0:
            print(f"\nEpoch {epoch+1:>4d} | ALL BATCHES SKIPPED ({num_skipped} skipped) — stopping early")
            return float("inf")

        avg_loss = total_loss / max(1, num_updates)
        dt = time.time() - t0
        loss_avgs = {k: v / max(1, num_updates) for k, v in loss_sums.items()}
        self._last_train_epoch_metrics = {
            f"train_{k}": v for k, v in loss_avgs.items() if k != "all"
        }
        self._last_train_epoch_metrics.update({
            "train_num_updates": num_updates,
            "train_num_skipped": num_skipped,
            "train_skip_ratio": skip_ratio,
        })

        # Build compact loss breakdown from epoch means, not the last batch.
        loss_parts = [f"train_loss={avg_loss:.6f}"]
        if num_updates > 0:
            sub = {
                k: loss_avgs[k]
                for k in ["noise_loss", "x0_weighted", "x0_ts",
                          "cticd_weighted", "moe_aux",
                          "cticd_total", "cticd_prior",
                          "cticd_lag_edge_density", "cticd_lag_prior_mean",
                          "cticd_lag_active_density", "cticd_feature_ratio"]
                if k in loss_avgs
            }
            if "noise_loss" in sub:
                loss_parts.append(f"noise={sub['noise_loss']:.4f}")
            if "x0_weighted" in sub:
                loss_parts.append(f"x0={sub['x0_weighted']:.4f}")
            if "cticd_weighted" in sub:
                loss_parts.append(f"cticd={sub['cticd_weighted']:.4f}")
            if "moe_aux" in sub:
                loss_parts.append(f"moe={sub['moe_aux']:.4f}")
            if "cticd_total" in sub:
                loss_parts.append(f"cticd_t={sub['cticd_total']:.4f}")
            if "cticd_prior" in sub:
                loss_parts.append(f"prior={sub['cticd_prior']:.4f}")
            if "cticd_lag_edge_density" in sub:
                loss_parts.append(f"lag_e={sub['cticd_lag_edge_density']:.3f}")
            if "cticd_lag_prior_mean" in sub:
                loss_parts.append(f"lag_p={sub['cticd_lag_prior_mean']:.3f}")
            if "cticd_lag_active_density" in sub:
                loss_parts.append(f"lag_a={sub['cticd_lag_active_density']:.3f}")
            if "cticd_feature_ratio" in sub:
                loss_parts.append(f"c_fr={sub['cticd_feature_ratio']:.3f}")
        loss_parts.append(f"lr={self.scheduler.get_lr():.2e}")
        loss_parts.append(f"{dt:.1f}s")
        loss_parts.append(f"skip={num_skipped}/{total_batches}")
        print(f"Epoch {epoch+1:>4d} | {' | '.join(loss_parts)}")

        self.writer.add_scalar("train_epoch/loss", avg_loss, epoch)
        self.writer.add_scalar("train_epoch/skip_ratio", skip_ratio, epoch)
        for k, v in loss_avgs.items():
            self.writer.add_scalar(f"train_epoch/{k}", v, epoch)
        return avg_loss

    @torch.no_grad()
    def _validate(
        self,
        epoch: int,
        save_best: bool = True,
        force_gen_metrics: bool = False,
        loader: DataLoader | None = None,
        prefix: str = "val",
    ) -> float:
        """Compute loss on a data loader (default: val_loader).

        Args:
            epoch: current epoch (for TensorBoard logging).
            save_best: whether to save a best checkpoint.
            force_gen_metrics: compute gen metrics regardless of interval.
            loader: data loader to evaluate (defaults to self.val_loader).
            prefix: log prefix ("val" or "test").
        """
        if loader is None:
            loader = self.val_loader

        self.model.eval()
        self._last_val_epoch_metrics = {}
        total_loss = 0.0
        num_batches = 0
        loss_sums = {}

        for batch in tqdm(loader, desc=f"Evaluating ({prefix})"):
            loss_dict = self.model(batch, is_train=False)
            if not torch.isfinite(loss_dict["all"]):
                print(f"WARNING: NaN/INF {prefix} loss, skipping batch")
                continue
            total_loss += loss_dict["all"].item()
            num_batches += 1
            for k, v in loss_dict.items():
                scalar = _scalar_float(v)
                if scalar is not None and np.isfinite(scalar):
                    loss_sums[k] = loss_sums.get(k, 0.0) + scalar

        avg_loss = total_loss / num_batches if num_batches else float("inf")
        self.writer.add_scalar(f"{prefix}/loss", avg_loss, epoch)
        loss_avgs = {k: v / max(1, num_batches) for k, v in loss_sums.items()}
        self._last_val_epoch_metrics = {
            f"{prefix}_{k}": v for k, v in loss_avgs.items() if k != "all"
        }
        self._last_val_epoch_metrics.update({
            f"{prefix}_num_batches": num_batches,
        })
        for k, v in loss_avgs.items():
            self.writer.add_scalar(f"{prefix}_epoch/{k}", v, epoch)
        print(f"         | {prefix}_loss  ={avg_loss:.6f}")

        # MSE_01/WAPE_01 every eval_gen_interval.
        # Eval-only forces generation metrics so a checkpoint validation is useful.
        do_gen_metrics = force_gen_metrics or (epoch + 1) % self.eval_gen_interval == 0
        gen_metrics = {}
        if do_gen_metrics:
            gen_metrics = self._compute_gen_metrics(epoch, loader=loader, prefix=prefix)

        # Merge gen metrics into the latest metrics_history entry
        if self.metrics_history:
            self.metrics_history[-1].update(self._last_val_epoch_metrics)
            if gen_metrics:
                self.metrics_history[-1].update(gen_metrics)
            with open(self.metrics_path, "w") as f:
                json.dump(_json_safe(self.metrics_history), f, indent=2)

        if save_best and avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self._save_checkpoint(epoch, val_loss=avg_loss, tag="best")
            print(f"         *** New best val loss: {avg_loss:.6f}")

        return avg_loss

    @torch.no_grad()
    def _compute_gen_metrics(
        self, epoch: int, loader: DataLoader | None = None, prefix: str = "val",
    ) -> dict:
        """Generate samples and compute MSE_01, WAPE_01, and CaTSG metrics.

        Args:
            epoch: current epoch (for TensorBoard logging).
            loader: data loader (defaults to self.val_loader).
            prefix: log prefix ("val" or "test").
        Returns dict of computed metrics.
        """
        if loader is None:
            loader = self.val_loader

        result = {}
        self.model.eval()
        dc = self.config["data"]
        ts_len = dc.get("time_interval", 24)
        representation = dc.get("representation", "gasf")
        if representation == "row_raster":
            from .ts_to_image import RowRasterEncoder
            H, W = RowRasterEncoder._optimal_hw(ts_len, self._row_raster_layout_patch_size())
            in_channels = self.config.get("diffusion", {}).get("in_channels", 1)
            image_shape = (in_channels, H, W)
        else:
            image_size = dc["image_size"]
            image_shape = (3, image_size, image_size)

        # Global min/max for T2S [0,1]-scale metric normalization.
        ds = loader.dataset
        g_min = ds.global_ts_min
        g_max = ds.global_ts_max
        g_range = max(g_max - g_min, 1e-8)

        # Collect both scales: [0,1] for T2S metrics, original for CaTSG metrics
        all_real_01, all_gen_01 = [], []
        all_real_orig, all_gen_orig = [], []
        all_texts = []

        for batch in loader:
            texts = batch.get("cap", None)
            images = batch["image"].to(self.device).float()
            ts_real = batch["ts"].to(self.device).float()
            ts_min = _squeeze_trailing_singletons(batch["ts_min"].to(self.device).float())
            ts_max = _squeeze_trailing_singletons(batch["ts_max"].to(self.device).float())

            ref_images = None
            if getattr(self.model, "is_multimodal", False):
                ref_images = self.model._batch_reference(batch, images, is_train=False)

            ccfg = self.config.get("condition", {})
            gen_imgs = self.model.generate(
                image_shape,
                texts,
                n_samples=1,
                ref_images=ref_images,
                guidance_scale=ccfg.get("text_guidance_scale", 1.0),
                image_guidance_scale=ccfg.get("image_guidance_scale", 1.0),
                interaction_guidance_scale=ccfg.get("interaction_guidance_scale", 0.0),
                cticd_enable_threshold=self.config.get("diffusion", {}).get("cticd_enable_threshold", 0.5),
            )
            gen_img = gen_imgs[0]

            decode_scale_mode = ccfg.get("decode_scale_mode", "global")
            if decode_scale_mode == "per_sample_oracle":
                dec_min, dec_max = ts_min, ts_max
            else:
                dec_min = torch.full_like(ts_min, float(g_min))
                dec_max = torch.full_like(ts_max, float(g_max))

            norm_params = NormParams(min_val=dec_min, max_val=dec_max, n_vars=1, original_length=ts_len)
            gen_ts = self.decoder.decode(gen_img, ts_len, norm_params)
            real_ts = denormalize_ts_batch(ts_real, ts_min, ts_max)

            if real_ts.shape != gen_ts.shape:
                raise RuntimeError(
                    f"Metric shape mismatch: real={tuple(real_ts.shape)}, gen={tuple(gen_ts.shape)}"
                )

            # T2S metrics: global [0,1] scale
            all_real_01.append(((real_ts - g_min) / g_range).cpu().numpy())
            all_gen_01.append(((gen_ts - g_min) / g_range).cpu().numpy())
            # CaTSG metrics: original scale
            all_real_orig.append(real_ts.cpu().numpy())
            all_gen_orig.append(gen_ts.cpu().numpy())
            if texts is not None:
                all_texts.extend(texts if isinstance(texts, list) else [texts])

        real_01 = np.concatenate(all_real_01, axis=0)
        gen_01 = np.concatenate(all_gen_01, axis=0)
        real_orig = np.concatenate(all_real_orig, axis=0)
        gen_orig = np.concatenate(all_gen_orig, axis=0)
        real_01_btd = _ensure_btd(real_01)
        gen_01_btd = _ensure_btd(gen_01)
        real_orig_btd = _ensure_btd(real_orig)
        gen_orig_btd = _ensure_btd(gen_orig)

        if self.config.get("save_eval_arrays", True):
            array_dir = os.path.join(self.config.get("log_dir", "."), "eval_arrays")
            os.makedirs(array_dir, exist_ok=True)
            tag = f"{prefix}_epoch{epoch + 1:04d}" if epoch >= 0 else prefix
            np.save(os.path.join(array_dir, f"{tag}_real_raw.npy"), real_orig_btd)
            np.save(os.path.join(array_dir, f"{tag}_gen_raw.npy"), gen_orig_btd)
            np.save(os.path.join(array_dir, f"{tag}_real_01.npy"), real_01_btd)
            np.save(os.path.join(array_dir, f"{tag}_gen_01.npy"), gen_01_btd)
            result["eval_array_dir"] = array_dir
            result["eval_array_tag"] = tag
            result["eval_array_shape"] = list(real_orig_btd.shape)

        # Encode text embeddings for J-FTSD
        cond_np = None
        if all_texts:
            try:
                with torch.no_grad():
                    text_emb = self.model.encode_text(all_texts)
                cond_np = text_emb.cpu().numpy()
            except Exception as e:
                print(f"WARNING: text encoding failed for J-FTSD: {e}")
                cond_np = None

        # T2S metrics on global [0,1] scale
        mse_01 = calc_mse(real_01_btd, gen_01_btd)
        wape_01 = calc_wape(real_01_btd, gen_01_btd)
        self.writer.add_scalar(f"{prefix}/MSE_01", mse_01, epoch)
        self.writer.add_scalar(f"{prefix}/WAPE_01", wape_01, epoch)
        result.update({"MSE_01": mse_01, "WAPE_01_macro": wape_01})

        msg = f"         | MSE_01={mse_01:.6f} | WAPE_01={wape_01:.4f}"

        # Unified metrics on original scale + [0,1] diagnostics
        try:
            from .evaluation.unified_metrics import compute_all_unified_metrics
            um = compute_all_unified_metrics(
                real_raw=real_orig_btd,
                gen_raw=gen_orig_btd,
                condition=cond_np,
                global_min=g_min,
                global_max=g_max,
                device=self.device,
                compute_01=True,
                compute_cfid=False,
                compute_jftsd=cond_np is not None,
            )
            # Log raw-scale metrics
            for key in ["MDD_raw_20", "KL_raw_flat", "out_of_range_rate_raw", "MMD_raw_rbf"]:
                if key in um:
                    self.writer.add_scalar(f"{prefix}/{key}", um[key], epoch)
                    result[key] = um[key]
            if um.get("J_FTSD") is not None:
                self.writer.add_scalar(f"{prefix}/J_FTSD", um["J_FTSD"], epoch)
                result["J_FTSD"] = um["J_FTSD"]
            msg += f" | MDD_raw={um['MDD_raw_20']:.4f} KL_raw={um['KL_raw_flat']:.4f} MMD_raw={um['MMD_raw_rbf']:.6f}"
            if um.get("J_FTSD") is not None:
                msg += f" J-FTSD={um['J_FTSD']:.4f}"

            # Log [0,1] diagnostics
            for key in ["MDD_01_20", "KL_01_flat", "MMD_01_rbf"]:
                if key in um:
                    self.writer.add_scalar(f"{prefix}/{key}", um[key], epoch)
                    result[key] = um[key]
        except Exception as e:
            print(f"         | CaTSG metrics skipped: {e}")

        print(msg)
        return result

    def _save_checkpoint(self, epoch: int, val_loss: float = None, tag: str = None):
        if tag is None:
            tag = f"epoch_{epoch+1}"
        ckpt_dir = os.path.join(self.config["save_dir"], "ckpts")
        os.makedirs(ckpt_dir, exist_ok=True)
        path = os.path.join(ckpt_dir, f"{tag}.pth")
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict() if self.optimizer else None,
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "epoch": epoch,
            "val_loss": val_loss,
            "global_step": self.global_step,
        }, path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="TIGER Training")

    # Paths
    p.add_argument("--data_dir", type=str, required=True, help="Path to dataset directory")
    p.add_argument("--config", type=str, default=None, help="JSON config file (overrides defaults)")
    p.add_argument("--save_dir", type=str, default=None)
    p.add_argument("--log_dir", type=str, default=None)
    p.add_argument("--model_path", type=str, default=None, help="Resume from checkpoint")

    # Data
    p.add_argument("--dataset_type", type=str, default=None,
                    choices=["weather_npy", "csv"])
    p.add_argument("--datasets", type=str, nargs="+", default=None,
                    help="T2S dataset names, e.g. --datasets ETTh1 traffic")
    p.add_argument("--time_interval", type=int, default=None,
                    choices=[24, 48, 96], help="T2S series length")
    p.add_argument("--image_size", type=int, default=None)
    p.add_argument("--n_fft", type=int, default=None)
    p.add_argument("--hop_length", type=int, default=None)

    # Training
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--warmup_steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)

    # Diffusion
    p.add_argument("--num_steps", type=int, default=None)
    p.add_argument("--channels", type=int, default=None)
    p.add_argument("--nheads", type=int, default=None)
    p.add_argument("--layers", type=int, default=None)
    p.add_argument("--n_var", type=int, default=None)
    p.add_argument("--multipatch_num", type=int, default=None)

    # Logging
    p.add_argument("--val_interval", type=int, default=None)
    p.add_argument("--display_interval", type=int, default=None)
    p.add_argument("--save_interval", type=int, default=None)
    p.add_argument(
        "--eval_only",
        action="store_true",
        help="Load --model_path and run validation/test metrics without training or saving checkpoints",
    )

    return p.parse_args()


def main():
    args = parse_args()

    # Build config: start from defaults / JSON, override with CLI
    config = load_config(args.config)
    config = apply_cli_overrides(config, args)

    # Auto-detect device
    if torch.cuda.is_available():
        config["device"] = "cuda:0"
    else:
        config["device"] = "cpu"
    set_seed(config["seed"])

    # Save effective config. Eval-only writes to log_dir so it cannot overwrite
    # the original training run's checkpoint config.
    if config.get("eval_only", False):
        os.makedirs(config["log_dir"], exist_ok=True)
        with open(os.path.join(config["log_dir"], "eval_config.json"), "w") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    else:
        os.makedirs(config["save_dir"], exist_ok=True)
        with open(os.path.join(config["save_dir"], "config.json"), "w") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

    trainer = TIGERTrainer(config)
    if config.get("eval_only", False):
        trainer.evaluate_only()
    else:
        trainer.train()


if __name__ == "__main__":
    main()
