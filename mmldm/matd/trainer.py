"""High-standard MATD trainer.

The trainer is intentionally thin when used with ``MATDModel``: the model owns
its forward pass and loss construction, while this class handles optimization,
AMP, EMA, scheduling, checkpointing and staged freezing.  It also supports the
legacy ``model.submodules`` dictionary by wrapping calls through an equivalent
model-like interface only when necessary.
"""
from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
try:
    from torch.amp import GradScaler, autocast
except ImportError:
    from torch.cuda.amp import GradScaler, autocast

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(it, **kw):
        return it

logger = logging.getLogger(__name__)


def _cfg_get(config: dict[str, Any], key: str, default: Any = None) -> Any:
    return config.get(key, default)


class EMA:
    """Exponential moving average over trainable parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        self.decay = decay
        self.shadow = {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}
        self.backup: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, p in model.named_parameters():
            if p.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    def apply(self, model: nn.Module) -> None:
        self.backup = {}
        for name, p in model.named_parameters():
            if p.requires_grad and name in self.shadow:
                self.backup[name] = p.detach().clone()
                p.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module) -> None:
        for name, p in model.named_parameters():
            if name in self.backup:
                p.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self) -> dict[str, Any]:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        self.decay = sd["decay"]
        self.shadow = sd["shadow"]


def _cosine_warmup_scheduler(optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.01) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return max(1e-8, float(step + 1) / max(warmup_steps, 1))
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class MATDTrainer:
    """Trainer for MATDModel with robust flat/nested config support."""

    def __init__(self, model: nn.Module, config: dict[str, Any], device: str = "cuda") -> None:
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.config = config
        if isinstance(model, dict):
            # Preserve legacy behaviour: pack submodules in ModuleDict.  Users are
            # encouraged to pass MATDModel directly for new training.
            self.model = nn.ModuleDict(model)
            self._legacy_dict_mode = True
        else:
            self.model = model
            self._legacy_dict_mode = False
        self.model.to(self.device)

        self.use_amp = bool(_cfg_get(config, "use_amp", True))
        self.grad_scaler = GradScaler(enabled=self.use_amp and self.device.type == "cuda")
        self.max_grad_norm = float(_cfg_get(config, "grad_clip", _cfg_get(config, "max_grad_norm", 1.0)))
        self.global_step = int(_cfg_get(config, "global_step", 0))
        self.log_interval = int(_cfg_get(config, "log_interval", 50))
        self.eval_interval = int(_cfg_get(config, "eval_interval", 10))
        self.best_metrics: dict[str, float] = {}
        self.best_sota_count: int = 0

        self.optimizer = self._build_optimizer()
        self.scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None
        self._build_scheduler(int(_cfg_get(config, "total_steps", 100000)))

        ema_cfg = config.get("ema", {}) if isinstance(config.get("ema", {}), dict) else {}
        self.use_ema = bool(ema_cfg.get("enabled", _cfg_get(config, "ema_enabled", True)))
        self.ema = EMA(self.model, decay=float(ema_cfg.get("decay", _cfg_get(config, "ema_decay", 0.9999)))) if self.use_ema else None

    def _build_optimizer(self) -> torch.optim.Optimizer:
        opt_cfg = self.config.get("optimizer", {}) if isinstance(self.config.get("optimizer", {}), dict) else {}
        lr = float(opt_cfg.get("lr", _cfg_get(self.config, "lr", 1e-4)))
        wd = float(opt_cfg.get("weight_decay", _cfg_get(self.config, "weight_decay", 0.01)))
        betas = tuple(opt_cfg.get("betas", _cfg_get(self.config, "betas", (0.9, 0.999))))
        params = [p for p in self.model.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=lr, betas=betas, weight_decay=wd)

    def _build_scheduler(self, total_steps: int) -> None:
        sched_cfg = self.config.get("scheduler", {}) if isinstance(self.config.get("scheduler", {}), dict) else {}
        warmup = int(sched_cfg.get("warmup_steps", _cfg_get(self.config, "warmup_steps", 1000)))
        min_lr_ratio = float(sched_cfg.get("min_lr_ratio", _cfg_get(self.config, "min_lr_ratio", 0.01)))
        self.scheduler = _cosine_warmup_scheduler(self.optimizer, warmup, total_steps, min_lr_ratio)

    def _set_trainable(self, trainable_names: Optional[set[str]]) -> None:
        if trainable_names is None:
            for p in self.model.parameters():
                p.requires_grad = True
            self.optimizer = self._build_optimizer()
            self._build_scheduler(int(_cfg_get(self.config, "total_steps", 100000)))
            return
        for name, module in self.named_components().items():
            req = name in trainable_names
            for p in module.parameters():
                p.requires_grad = req
        self.optimizer = self._build_optimizer()
        self._build_scheduler(int(_cfg_get(self.config, "total_steps", 100000)))

    def named_components(self) -> dict[str, nn.Module]:
        if hasattr(self.model, "submodules") and isinstance(getattr(self.model, "submodules"), dict):
            return getattr(self.model, "submodules")
        if isinstance(self.model, nn.ModuleDict):
            return dict(self.model.items())
        return {"model": self.model}

    def _forward(self, batch: tuple, stage: int = 3, meta_override: Optional[torch.Tensor] = None) -> dict[str, Any]:
        if self._legacy_dict_mode:
            raise RuntimeError("Legacy dict-mode forward is no longer recommended. Pass a MATDModel instance to MATDTrainer.")
        x0, texts = batch[:2]
        x0 = x0.to(self.device)
        if hasattr(self.model, "forward_train"):
            return self.model.forward_train(x0, list(texts), meta_override=meta_override, stage=stage)
        raise RuntimeError("model must implement forward_train(x0, texts, meta_override=None)")

    def train_step(self, batch: tuple, stage: int = 3, meta_override: Optional[torch.Tensor] = None) -> dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=self.use_amp and self.device.type == "cuda"):
            out = self._forward(batch, stage=stage, meta_override=meta_override)
            loss = out["loss"] if "loss" in out else out["loss_total"]
        terms = out.get("loss_terms", {"loss_total": loss})
        metrics = {k: (v.detach().item() if torch.is_tensor(v) else float(v)) for k, v in terms.items()}
        if not torch.isfinite(loss.detach()):
            bad_terms = {k: v for k, v in metrics.items() if not math.isfinite(v)}
            logger.debug("stage=%d step=%d non-finite loss; skipping optimizer step; bad_terms=%s", stage, self.global_step, bad_terms)
            self.optimizer.zero_grad(set_to_none=True)
            if self.grad_scaler.is_enabled():
                self.grad_scaler.update(max(self.grad_scaler.get_scale() * 0.5, 1.0))
            metrics["skipped_step"] = 1.0
            return metrics
        self.grad_scaler.scale(loss).backward()
        self.grad_scaler.unscale_(self.optimizer)
        grad_norm = torch.tensor(0.0, device=self.device)
        if self.max_grad_norm > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        if torch.is_tensor(grad_norm) and not torch.isfinite(grad_norm.detach()):
            logger.debug("stage=%d step=%d non-finite grad_norm=%s; skipping optimizer step", stage, self.global_step, grad_norm.detach().item())
            self.optimizer.zero_grad(set_to_none=True)
            if self.grad_scaler.is_enabled():
                self.grad_scaler.update(max(self.grad_scaler.get_scale() * 0.5, 1.0))
            metrics["grad_norm"] = float("nan")
            metrics["skipped_step"] = 1.0
            return metrics
        old_scale = self.grad_scaler.get_scale() if self.grad_scaler.is_enabled() else 1.0
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()
        new_scale = self.grad_scaler.get_scale() if self.grad_scaler.is_enabled() else old_scale
        optimizer_stepped = (not self.grad_scaler.is_enabled()) or new_scale >= old_scale
        if self.scheduler is not None and self.global_step > 0 and optimizer_stepped:
            self.scheduler.step()
        if self.ema is not None and optimizer_stepped:
            self.ema.update(self.model)
        self.global_step += 1
        metrics["grad_norm"] = float(grad_norm.detach().item()) if torch.is_tensor(grad_norm) else float(grad_norm)
        metrics["skipped_step"] = 0.0 if optimizer_stepped else 1.0
        return metrics

    def train_stage(self, dataloader: torch.utils.data.DataLoader, epochs: int, stage: int = 3, val_dataloader: Optional[torch.utils.data.DataLoader] = None, evaluator: Optional[Any] = None, save_dir: Optional[str] = None) -> list[dict[str, float]]:
        # Stage freezing.  Full model training remains default for stages 3/4.
        if stage == 1:
            self._set_trainable({"encoder", "decoder"})
        elif stage == 2:
            self._set_trainable({"text_encoder", "null_encoder", "planner", "slot_extractor", "injector"})
        else:
            self._set_trainable(None)
        history: list[dict[str, float]] = []
        for epoch in range(epochs):
            accum: dict[str, float] = {}
            n = 0
            skipped = 0
            pbar = tqdm(dataloader, desc=f"stage{stage}-epoch{epoch}")
            for batch in pbar:
                metrics = self.train_step(batch, stage=stage)
                for k, v in metrics.items():
                    accum[k] = accum.get(k, 0.0) + v
                n += 1
                if metrics.get("skipped_step", 0.0) > 0:
                    skipped += 1
                if hasattr(pbar, "set_postfix") and self.global_step % self.log_interval == 0:
                    pbar.set_postfix(loss_total=f"{metrics.get('loss_total', 0.0):.5f}")
                if self.global_step % self.log_interval == 0:
                    logger.info("stage=%d step=%d loss_total=%.5f", stage, self.global_step, metrics.get("loss_total", 0.0))
            avg = {k: v / max(n, 1) for k, v in accum.items()}
            avg["epoch"] = float(epoch)
            history.append(avg)
            logger.info("stage=%d epoch=%d train_loss=%.5f", stage, epoch, avg.get("loss_total", 0.0))
            if skipped > 0:
                logger.warning("stage=%d epoch=%d %d/%d batches skipped (non-finite loss/grad)", stage, epoch, skipped, n)
            if val_dataloader is not None:
                val = self.validate(val_dataloader, stage=stage)
                logger.info("stage=%d epoch=%d val_loss=%.5f", stage, epoch, val.get("loss_total", 0.0))
            # Stage-4: run evaluation every eval_interval epochs
            if stage == 4 and evaluator is not None and (epoch + 1) % self.eval_interval == 0:
                eval_results = evaluator.evaluate()
                evaluator.print_results(eval_results)
                sota_count = self._check_sota(eval_results, save_dir)
                logger.info("stage=%d epoch=%d eval SOTA count=%d", stage, epoch, sota_count)
        return history

    def train_all_stages(self, train_loaders: dict[int, torch.utils.data.DataLoader], epochs_per_stage: dict[int, int], val_loaders: Optional[dict[int, torch.utils.data.DataLoader]] = None, save_dir: Optional[str] = None, mix_ratio: float = 0.5, evaluator: Optional[Any] = None) -> dict[int, list[dict[str, float]]]:
        hist: dict[int, list[dict[str, float]]] = {}
        for stage in (1, 2, 3, 4):
            if stage not in train_loaders:
                continue
            hist[stage] = self.train_stage(train_loaders[stage], epochs_per_stage.get(stage, 1), stage=stage, val_dataloader=(val_loaders or {}).get(stage), evaluator=evaluator if stage == 4 else None, save_dir=save_dir)
            if save_dir is not None:
                self.save_checkpoint(os.path.join(save_dir, f"stage{stage}.pt"))
        return hist

    @torch.no_grad()
    def validate(self, dataloader: torch.utils.data.DataLoader, stage: int = 3) -> dict[str, float]:
        if self.ema is not None:
            self.ema.apply(self.model)
        self.model.eval()
        accum: dict[str, float] = {}
        n = 0
        for batch in dataloader:
            x0, texts = batch[:2]
            x0 = x0.to(self.device)
            with autocast(enabled=self.use_amp and self.device.type == "cuda"):
                out = self.model.forward_train(x0, list(texts), stage=stage)
            for k, v in out.get("loss_terms", {}).items():
                if torch.is_tensor(v):
                    accum[k] = accum.get(k, 0.0) + float(v.detach().item())
            n += 1
        if self.ema is not None:
            self.ema.restore(self.model)
        return {k: v / max(n, 1) for k, v in accum.items()}

    def _check_sota(self, results: dict[str, Any], save_dir: Optional[str] = None) -> int:
        """Check if current metrics are SOTA and save best checkpoint."""
        # Metrics where lower is better
        lower_better = {"MSE", "WAPE", "MDD", "KL", "MMD", "C-FID"}
        # Metrics where higher is better
        higher_better = {"MRR"}
        sota_count = 0
        improved = []
        for metric in lower_better | higher_better:
            val = results.get(metric)
            if val is None or val != val:  # skip NaN
                continue
            if metric in lower_better:
                if metric not in self.best_metrics or val < self.best_metrics[metric]:
                    self.best_metrics[metric] = val
                    improved.append(metric)
            else:
                if metric not in self.best_metrics or val > self.best_metrics[metric]:
                    self.best_metrics[metric] = val
                    improved.append(metric)
        # Count how many metrics are at their best
        current_sota = 0
        for metric in lower_better | higher_better:
            val = results.get(metric)
            if val is None or val != val:
                continue
            if metric in lower_better and val <= self.best_metrics.get(metric, float("inf")):
                current_sota += 1
            elif metric in higher_better and val >= self.best_metrics.get(metric, float("-inf")):
                current_sota += 1
        if improved:
            logger.info("Improved metrics: %s", improved)
        # Save best checkpoint if >= 3 metrics are SOTA
        if current_sota >= 3 and current_sota > self.best_sota_count:
            self.best_sota_count = current_sota
            if save_dir is not None:
                best_path = os.path.join(save_dir, "best.pt")
                self.save_checkpoint(best_path)
                logger.info("Saved best checkpoint (%d SOTA metrics) to %s", current_sota, best_path)
        return current_sota

    def save_checkpoint(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        state = {
            "global_step": self.global_step,
            "config": self.config,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "scaler": self.grad_scaler.state_dict(),
            "ema": self.ema.state_dict() if self.ema is not None else None,
        }
        torch.save(state, path)

    def load_checkpoint(self, path: str, strict: bool = True) -> None:
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model"], strict=strict)
        if "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])
        if self.scheduler is not None and state.get("scheduler") is not None:
            self.scheduler.load_state_dict(state["scheduler"])
        if state.get("scaler") is not None:
            self.grad_scaler.load_state_dict(state["scaler"])
        if self.ema is not None and state.get("ema") is not None:
            self.ema.load_state_dict(state["ema"])
        self.global_step = int(state.get("global_step", 0))
