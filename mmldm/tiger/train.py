"""TIGER Training Pipeline.

Trains the Text+Image Guided diffusion model for time-series generation.

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

        "diffusion": {
            "num_steps": 50,
            "beta_start": 0.0001,
            "beta_end": 0.5,
            "schedule": "quad",
            "channels": 64,
            "nheads": 8,
            "layers": 8,
            "n_var": 16,
            "multipatch_num": 1,
            "base_patch": 4,
            "patch_scale": 2,
            "diffusion_embedding_dim": 64,
            "in_channels": 3,
            "condition_type": "adaLN",
            "attention_mask_type": "parallel",
        },

        "condition": {
            "cond_mode": "text+image",   # "text+image" | "text_only" | "image_only"
            "image_encoder_type": "cnn",
            "num_stages": 4,
            "cfg_dropout": 0.3,          # 30% prob to drop text for CFG training
            "text": {
                "pretrain_model_path": "openai/clip-vit-base-patch32",
                "pretrain_model_dim": 512,
                "textemb_hidden_dim": 256,
                "text_emb": 128,
            },
            "image": {
                "pretrain_model_path": "openai/clip-vit-base-patch32",
                "pretrain_model_dim": 768,
                "imageemb_hidden_dim": 256,
                "image_emb": 128,
                "device": "cuda:0",
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

        os.makedirs(config["save_dir"], exist_ok=True)
        os.makedirs(config["log_dir"], exist_ok=True)

        self._init_data()
        self._init_model()
        self._init_opt()
        self._init_logging()

        self.best_val_loss = float("inf")
        self.global_step = 0

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
        train_ds = TIGERDataset(split="train", **common)
        val_ds = TIGERDataset(split="valid" if dc["dataset_type"] == "weather_npy" else "test",
                              **common)

        collate = TIGERCollateFn()
        self.train_loader = DataLoader(
            train_ds, batch_size=self.config["batch_size"],
            shuffle=True, collate_fn=collate, num_workers=0,
            drop_last=True,
        )
        self.val_loader = DataLoader(
            val_ds, batch_size=self.config["batch_size"],
            shuffle=False, collate_fn=collate, num_workers=0,
        )
        self.train_dataset = train_ds

    # ---- model --------------------------------------------------------------

    def _init_model(self):
        model_config = {
            "device": self.device,
            "diffusion": self.config["diffusion"],
            "condition": self.config["condition"],
        }
        self.model = TIGERGenerator(model_config)

        if self.config.get("model_path"):
            print(f"Loading pretrained model from {self.config['model_path']}")
            state = torch.load(self.config["model_path"], map_location=self.device)
            self.model.load_state_dict(state, strict=False)

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
        print(f"Starting training for {self.n_epochs} epochs")
        print(f"  Train samples: {len(self.train_loader.dataset)}")
        print(f"  Val samples:   {len(self.val_loader.dataset)}")
        print(f"  Batch size:    {self.config['batch_size']}")
        print(f"  Device:        {self.device}")

        for epoch in range(self.n_epochs):
            train_loss = self._train_epoch(epoch)

            if (epoch + 1) % self.val_interval == 0:
                val_loss = self._validate(epoch)
                self._save_checkpoint(epoch, val_loss)

            if (epoch + 1) % self.save_interval == 0:
                self._save_checkpoint(epoch, tag=f"epoch_{epoch+1}")

        self.writer.close()
        print("Training complete.")

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        t0 = time.time()

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}", leave=False)
        for batch in pbar:
            self.optimizer.zero_grad()
            loss_dict = self.model(batch, is_train=True)
            loss_dict["all"].backward()

            # Gradient clipping
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()
            self.scheduler.step()
            self.global_step += 1

            total_loss += loss_dict["all"].item()
            pbar.set_postfix(loss=f"{loss_dict['all'].item():.4f}", lr=f"{self.scheduler.get_lr():.2e}")

            # Log per-step
            for k, v in loss_dict.items():
                self.writer.add_scalar(f"train_step/{k}", v.item(), self.global_step)
            self.writer.add_scalar("train_step/lr", self.scheduler.get_lr(), self.global_step)

        avg_loss = total_loss / max(1, len(self.train_loader))
        dt = time.time() - t0

        if (epoch + 1) % self.display_interval == 0:
            print(f"Epoch {epoch+1:>4d} | train_loss={avg_loss:.6f} | lr={self.scheduler.get_lr():.2e} | {dt:.1f}s")

        self.writer.add_scalar("train_epoch/loss", avg_loss, epoch)
        return avg_loss

    @torch.no_grad()
    def _validate(self, epoch: int) -> float:
        self.model.eval()
        total_loss = 0.0

        for batch in tqdm(self.val_loader, desc="Validating", leave=False):
            loss_dict = self.model(batch, is_train=False)
            total_loss += loss_dict["all"].item()

        avg_loss = total_loss / max(1, len(self.val_loader))
        self.writer.add_scalar("val/loss", avg_loss, epoch)
        print(f"         | val_loss  ={avg_loss:.6f}")

        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self._save_checkpoint(epoch, tag="best")
            print(f"         *** New best val loss: {avg_loss:.6f}")

        return avg_loss

    def _save_checkpoint(self, epoch: int, val_loss: float = None, tag: str = None):
        if tag is None:
            tag = f"epoch_{epoch+1}"
        ckpt_dir = os.path.join(self.config["save_dir"], "ckpts")
        os.makedirs(ckpt_dir, exist_ok=True)
        path = os.path.join(ckpt_dir, f"{tag}.pth")
        torch.save(self.model.state_dict(), path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="TIGER Training")

    # Paths
    p.add_argument("--data_dir", type=str, required=True, help="Path to dataset directory")
    p.add_argument("--config", type=str, default=None, help="JSON config file (overrides defaults)")
    p.add_argument("--save_dir", type=str, default="./checkpoints/tiger")
    p.add_argument("--log_dir", type=str, default="./runs/tiger")
    p.add_argument("--model_path", type=str, default="", help="Resume from checkpoint")

    # Data
    p.add_argument("--dataset_type", type=str, default=None,
                    choices=["weather_npy", "csv"])
    p.add_argument("--datasets", type=str, nargs="+", default=None,
                    help="T2S dataset names, e.g. --datasets ETTh1 traffic")
    p.add_argument("--time_interval", type=int, default=None,
                    choices=[24, 48, 96], help="T2S series length")
    p.add_argument("--image_size", type=int, default=64)
    p.add_argument("--n_fft", type=int, default=64)
    p.add_argument("--hop_length", type=int, default=8)

    # Training
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)

    # Diffusion
    p.add_argument("--num_steps", type=int, default=50)
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--nheads", type=int, default=8)
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--n_var", type=int, default=16)
    p.add_argument("--multipatch_num", type=int, default=1)

    # Condition
    p.add_argument("--use_text", action="store_true", default=True)
    p.add_argument("--no_text", dest="use_text", action="store_false")
    p.add_argument("--image_encoder_type", type=str, default="vit",
                    choices=["cnn", "clip", "vit"])

    # Logging
    p.add_argument("--val_interval", type=int, default=10)
    p.add_argument("--display_interval", type=int, default=10)
    p.add_argument("--save_interval", type=int, default=50)

    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    # Build config: start from defaults / JSON, override with CLI
    config = load_config(args.config)
    config.update({
        "data_dir": args.data_dir,
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
    })
    if args.dataset_type is not None:
        config["data"]["dataset_type"] = args.dataset_type
    if args.datasets is not None:
        config["data"]["datasets"] = args.datasets
    if args.time_interval is not None:
        config["data"]["time_interval"] = args.time_interval
    config["data"].update({
        "image_size": args.image_size,
        "n_fft": args.n_fft,
        "hop_length": args.hop_length,
    })
    config["diffusion"].update({
        "num_steps": args.num_steps,
        "channels": args.channels,
        "nheads": args.nheads,
        "layers": args.layers,
        "n_var": args.n_var,
        "multipatch_num": args.multipatch_num,
    })
    config["condition"]["use_text"] = args.use_text
    config["condition"]["image_encoder_type"] = args.image_encoder_type

    # Auto-detect device
    if torch.cuda.is_available():
        config["device"] = "cuda:0"
    else:
        config["device"] = "cpu"

    # Save effective config
    os.makedirs(config["save_dir"], exist_ok=True)
    with open(os.path.join(config["save_dir"], "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    trainer = TIGERTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
