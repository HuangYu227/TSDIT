"""Observe-only loss balancing diagnostics for MATD training.

This module deliberately does not alter the optimization objective.  It probes
gradient norms and pairwise gradient cosine similarities for the MATD loss
groups so training logs can reveal scale imbalance and gradient conflict.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import nn


GROUP_AGG_KEYS = {
    "diffusion_bridge": "loss_diffusion_bridge",
    "patch_field": "loss_patch_field",
    "text_layout": "loss_text_layout",
    "mechanism": "loss_mechanism",
}


@dataclass
class LossBalanceConfig:
    """Configuration for observe-only gradient diagnostics."""

    enabled: bool = False
    interval: int = 50
    probe_components: tuple[str, ...] | None = None

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any] | None) -> "LossBalanceConfig":
        if not isinstance(config, Mapping):
            return cls()
        probe = config.get("probe_components")
        if probe is None:
            probe_components = None
        elif isinstance(probe, str):
            probe_components = (probe,)
        else:
            probe_components = tuple(str(item) for item in probe)
        return cls(
            enabled=bool(config.get("enabled", False)),
            interval=max(1, int(config.get("interval", 50))),
            probe_components=probe_components,
        )

    def should_probe(self, global_step: int) -> bool:
        return self.enabled and global_step % self.interval == 0


def _select_parameters(
    model: nn.Module,
    components: Sequence[str] | None = None,
) -> list[torch.nn.Parameter]:
    selected: list[torch.nn.Parameter] = []
    filters = tuple(components or ())
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if filters and not any(f in name for f in filters):
            continue
        selected.append(param)
    return selected


def _flatten_grads(
    loss: torch.Tensor,
    params: Sequence[torch.nn.Parameter],
) -> torch.Tensor | None:
    if not torch.is_tensor(loss) or not loss.requires_grad or not params:
        return None
    grads = torch.autograd.grad(
        loss,
        params,
        retain_graph=True,
        allow_unused=True,
        create_graph=False,
    )
    flat_parts: list[torch.Tensor] = []
    for grad, param in zip(grads, params):
        if grad is None:
            flat_parts.append(torch.zeros(param.numel(), device=param.device, dtype=torch.float32))
        else:
            flat_parts.append(grad.detach().float().reshape(-1))
    if not flat_parts:
        return None
    return torch.cat(flat_parts)


def _group_loss(
    loss_dicts: Mapping[str, Mapping[str, torch.Tensor]],
    group_name: str,
) -> torch.Tensor | None:
    sub = loss_dicts.get(group_name)
    if not isinstance(sub, Mapping):
        return None
    key = GROUP_AGG_KEYS[group_name]
    value = sub.get(key)
    return value if torch.is_tensor(value) else None


class GradientConflictMonitor:
    """Compute gradient norm and cosine diagnostics for MATD loss groups."""

    group_names = ("diffusion_bridge", "patch_field", "text_layout", "mechanism")

    def compute(
        self,
        loss_dicts: Mapping[str, Mapping[str, torch.Tensor]],
        model: nn.Module,
        components: Sequence[str] | None = None,
    ) -> dict[str, float]:
        """Return observe-only gradient diagnostics.

        Args:
            loss_dicts: Nested loss dictionary returned by ``MATDModel.forward_train``.
            model: Model whose trainable parameters are probed.
            components: Optional substrings used to restrict probed parameter names.

        Returns:
            Flat dictionary with ``grad_group_*`` and ``grad_cos_*`` entries.
        """
        params = _select_parameters(model, components)
        out: dict[str, float] = {
            "grad_probe_param_count": float(len(params)),
            "grad_probe_group_count": 0.0,
        }
        vectors: dict[str, torch.Tensor] = {}
        for group in self.group_names:
            loss = _group_loss(loss_dicts, group)
            vec = _flatten_grads(loss, params) if loss is not None else None
            if vec is None:
                out[f"grad_group_{group}_norm"] = 0.0
                out[f"grad_group_{group}_has_grad"] = 0.0
                continue
            norm = torch.linalg.vector_norm(vec)
            out[f"grad_group_{group}_norm"] = float(norm.detach().cpu().item())
            has_grad = bool(torch.isfinite(norm) and norm > 0)
            out[f"grad_group_{group}_has_grad"] = float(has_grad)
            if has_grad:
                vectors[group] = vec
        out["grad_probe_group_count"] = float(len(vectors))

        names = list(self.group_names)
        for i, left in enumerate(names):
            for right in names[i + 1 :]:
                key = f"grad_cos_{left}__{right}"
                if left not in vectors or right not in vectors:
                    out[key] = 0.0
                    continue
                l_vec = vectors[left]
                r_vec = vectors[right]
                denom = torch.linalg.vector_norm(l_vec) * torch.linalg.vector_norm(r_vec)
                if not torch.isfinite(denom) or denom <= 0:
                    out[key] = 0.0
                else:
                    cos = torch.dot(l_vec, r_vec) / denom.clamp_min(1e-12)
                    out[key] = float(cos.clamp(-1.0, 1.0).detach().cpu().item())
        return out
