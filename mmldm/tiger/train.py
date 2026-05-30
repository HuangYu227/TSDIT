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
            "channels": 64,
            "nheads": 8,
            "layers": 8,
            "n_var": 16,
            "multipatch_num": 4,
            "base_patch": 4,
            "patch_scale": 2,
            "diffusion_embedding_dim": 64,
            "in_channels": 3,
            "condition_type": "adaLN",
            "attention_mask_type": "parallel",
        },

        "condition": {
            "cond_mode": "text_only",
            "num_stages": 4,
            "cfg_dropout": 0.3,          # 30% prob to drop text for CFG training
            "text": {
                "pretrain_model_path": "openai/clip-vit-base-patch32",
                "pretrain_model_dim": 512,
                "textemb_hidden_dim": 256,
                "text_emb": 128,
            },
        },

        "data": {
            "dataset_type": "weather_npy",
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


def calc_mape(real: np.ndarray, gen: np.ndarray, eps: float = 1e-3) -> float:
    """MAPE between real and generated time series. Both (B, T)."""
    return float(np.mean(np.abs(real - gen) / np.maximum(np.abs(real), eps)))


def calc_wape(real: np.ndarray, gen: np.ndarray) -> float:
    """WAPE: sum(|real - gen|) / sum(|real|). T2S style."""
    return float(np.sum(np.abs(real - gen)) / (np.sum(np.abs(real)) + 1e-8))


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
        self.eval_gen_interval = config.get("eval_gen_interval", 5)

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
        self.decoder = ImageToTSDecoder(
            mode="gasf", n_fft=dc["n_fft"], hop_length=dc["hop_length"]
        ).to(self.device)

        self.best_val_loss = float("inf")
        self.global_step = 0

        # Metrics history (saved to JSON)
        self.metrics_history: list = []
        self.metrics_path = os.path.join(config.get("log_dir", "."), "metrics.json")

    # ---- data ---------------------------------------------------------------

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
        )
        val_ds = TIGERDataset(split="valid" if dc["dataset_type"] == "weather_npy" else "test",
                              **common)

        collate = TIGERCollateFn()
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
        self.val_loader = DataLoader(
            val_ds, batch_size=self.config["batch_size"],
            shuffle=False, collate_fn=collate, num_workers=0,
        )

    # ---- model --------------------------------------------------------------

    def _init_model(self):
        model_config = {
            "device": self.device,
            "diffusion": self.config["diffusion"],
            "condition": self.config["condition"],
        }
        # Pass top-level csa_moe / cticd configs so dit_model can read them
        if "csa_moe" in self.config:
            model_config["diffusion"]["csa_moe"] = self.config["csa_moe"]
        if "cticd" in self.config:
            model_config["diffusion"]["cticd"] = self.config["cticd"]
        # Pass image_size for MoE grid computation
        if "data" in self.config and "image_size" in self.config["data"]:
            model_config["diffusion"]["image_size"] = self.config["data"]["image_size"]
        self.model = TIGERGenerator(model_config)

        if self.config.get("model_path"):
            print(f"Loading pretrained model from {self.config['model_path']}")
            ckpt = torch.load(self.config["model_path"], map_location=self.device)
            # Support both old (state_dict only) and new (dict) checkpoint formats
            if isinstance(ckpt, dict) and "model" in ckpt:
                self.model.load_state_dict(ckpt["model"], strict=False)
            else:
                self.model.load_state_dict(ckpt, strict=False)

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

            if (epoch + 1) % self.val_interval == 0:
                val_loss = self._validate(epoch)
                epoch_metrics["val_loss"] = val_loss
                self._save_checkpoint(epoch, val_loss)

            if (epoch + 1) % self.save_interval == 0:
                self._save_checkpoint(epoch, tag=f"epoch_{epoch+1}")

            self.metrics_history.append(epoch_metrics)
            with open(self.metrics_path, "w") as f:
                json.dump(self.metrics_history, f, indent=2)

        self.writer.close()
        print("Training complete.")

    def evaluate_only(self):
        """Evaluate a checkpoint without training or writing checkpoint files."""
        if not self.config.get("model_path"):
            raise ValueError("--eval_only requires --model_path")

        print("Starting eval-only run")
        print(f"  Eval samples: {len(self.val_loader.dataset)}")
        print(f"  Batch size:   {self.config['batch_size']}")
        print(f"  Device:       {self.device}")
        print(f"  Checkpoint:   {self.config['model_path']}")

        val_loss = self._validate(
            epoch=0,
            save_best=False,
            force_gen_metrics=True,
        )
        self.writer.close()
        print(f"Eval complete. val_loss={val_loss:.6f}")
        return val_loss

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        num_updates = 0
        num_skipped = 0
        t0 = time.time()

        # Early stopping config
        early_stop_lr = self.config.get("early_stop_lr", 1e-6)
        max_skip_ratio = self.config.get("max_skip_ratio", 0.5)

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            self.optimizer.zero_grad()
            loss_dict = self.model(batch, is_train=True)
            loss = loss_dict["all"]
            if not torch.isfinite(loss) or loss.item() < 1e-7:
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
            pbar.set_postfix(loss=f"{loss_dict['all'].item():.4f}", grad=f"{grad_norm:.2f}", lr=f"{self.scheduler.get_lr():.2e}")

            # Log per-step
            for k, v in loss_dict.items():
                self.writer.add_scalar(f"train_step/{k}", v.item(), self.global_step)
            self.writer.add_scalar("train_step/lr", self.scheduler.get_lr(), self.global_step)

        # Check skip ratio at end of epoch
        total_batches = num_updates + num_skipped
        skip_ratio = num_skipped / total_batches if total_batches > 0 else 0
        if skip_ratio > max_skip_ratio:
            print(f"\nWARNING: High skip ratio {skip_ratio:.1%} ({num_skipped}/{total_batches}) at epoch {epoch+1}")

        avg_loss = total_loss / max(1, num_updates)
        dt = time.time() - t0
        print(f"Epoch {epoch+1:>4d} | train_loss={avg_loss:.6f} | lr={self.scheduler.get_lr():.2e} | {dt:.1f}s | skipped={num_skipped}/{total_batches}")

        self.writer.add_scalar("train_epoch/loss", avg_loss, epoch)
        self.writer.add_scalar("train_epoch/skip_ratio", skip_ratio, epoch)
        return avg_loss

    @torch.no_grad()
    def _validate(
        self,
        epoch: int,
        save_best: bool = True,
        force_gen_metrics: bool = False,
    ) -> float:
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(self.val_loader, desc="Validating"):
            loss_dict = self.model(batch, is_train=False)
            if not torch.isfinite(loss_dict["all"]):
                print("WARNING: NaN/INF validation loss, skipping batch")
                continue
            total_loss += loss_dict["all"].item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches else float("inf")
        self.writer.add_scalar("val/loss", avg_loss, epoch)
        print(f"         | val_loss  ={avg_loss:.6f}")

        # MSE/MAPE every eval_gen_interval.
        # Eval-only forces generation metrics so a checkpoint validation is useful.
        do_gen_metrics = force_gen_metrics or (epoch + 1) % self.eval_gen_interval == 0
        gen_metrics = {}
        if do_gen_metrics:
            gen_metrics = self._compute_gen_metrics(epoch)

        # Merge gen metrics into the latest metrics_history entry
        if self.metrics_history and gen_metrics:
            self.metrics_history[-1].update(gen_metrics)
            with open(self.metrics_path, "w") as f:
                json.dump(self.metrics_history, f, indent=2)

        if save_best and avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self._save_checkpoint(epoch, tag="best")
            print(f"         *** New best val loss: {avg_loss:.6f}")

        return avg_loss

    @torch.no_grad()
    def _compute_gen_metrics(self, epoch: int) -> dict:
        """Generate samples and compute MSE, MAPE, and CaTSG metrics.
        Returns dict of computed metrics."""
        result = {}
        self.model.eval()
        dc = self.config["data"]
        ts_len = dc.get("time_interval", 24)
        image_size = dc["image_size"]
        image_shape = (3, image_size, image_size)

        # Global min/max for scale-leakage-free comparison
        # Use val_loader.dataset (always available) instead of train_loader
        # (which is None in eval-only mode)
        ds = self.val_loader.dataset
        g_min = ds.global_ts_min
        g_max = ds.global_ts_max
        g_range = max(g_max - g_min, 1e-8)

        # Collect both scales: [0,1] for T2S metrics, original for CaTSG metrics
        all_real_01, all_gen_01 = [], []
        all_real_orig, all_gen_orig = [], []
        all_texts = []

        for batch in self.val_loader:
            texts = batch.get("cap", None)
            ts_real = batch["ts"].to(self.device).float()
            ts_min = _squeeze_trailing_singletons(batch["ts_min"].to(self.device).float())
            ts_max = _squeeze_trailing_singletons(batch["ts_max"].to(self.device).float())

            gen_imgs = self.model.generate(image_shape, texts, n_samples=1)
            gen_img = gen_imgs[0]
            norm_params = NormParams(min_val=ts_min, max_val=ts_max, n_vars=1, original_length=ts_len)
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
        mse = calc_mse(real_01, gen_01)
        mape = calc_mape(real_01, gen_01)
        wape = calc_wape(real_01, gen_01)
        self.writer.add_scalar("val/MSE", mse, epoch)
        self.writer.add_scalar("val/MAPE", mape, epoch)
        self.writer.add_scalar("val/WAPE", wape, epoch)
        result.update({"MSE": mse, "MAPE": mape, "WAPE": wape})

        msg = f"         | MSE={mse:.6f} | MAPE={mape:.4f} | WAPE={wape:.4f}"

        # CaTSG metrics on original scale (matches CaTSG paper evaluation)
        try:
            from .evaluation.catsg_metrics import compute_all_catsg_metrics
            catsg = compute_all_catsg_metrics(real_orig, gen_orig, cond=cond_np, device=self.device)
            for k, v in catsg.items():
                self.writer.add_scalar(f"val/{k}", v, epoch)
                result[k] = v
            msg += f" | MDD={catsg['MDD']:.4f} KL={catsg['KL']:.4f} MMD={catsg['MMD']:.6f}"
            if "J-FTSD" in catsg:
                msg += f" J-FTSD={catsg['J-FTSD']:.4f}"
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
