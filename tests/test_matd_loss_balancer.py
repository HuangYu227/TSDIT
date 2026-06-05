import math

import torch
from torch import nn

from mmldm.matd.loss_balancer import GradientConflictMonitor, LossBalanceConfig


class ToyMATDHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(3, 4)
        self.decoder = nn.Linear(4, 2)
        self.unused = nn.Parameter(torch.ones(5))

    def forward(self, x):
        h = torch.tanh(self.encoder(x))
        return self.decoder(h), h


def _loss_dicts(model, x):
    y, h = model(x)
    target = torch.zeros_like(y)
    return {
        "diffusion_bridge": {"loss_diffusion_bridge": torch.mean((y - target) ** 2)},
        "patch_field": {"loss_patch_field": torch.mean(torch.abs(y))},
        "text_layout": {"loss_text_layout": torch.mean(h ** 2)},
        "mechanism": {"loss_mechanism": torch.mean((h[:, 1:] - h[:, :-1]) ** 2)},
    }


def test_loss_balance_config_from_mapping_defaults_and_probe_components():
    cfg = LossBalanceConfig.from_mapping(
        {"enabled": True, "interval": 7, "probe_components": ["encoder"]}
    )

    assert cfg.enabled is True
    assert cfg.interval == 7
    assert cfg.probe_components == ("encoder",)
    assert cfg.should_probe(14)
    assert not cfg.should_probe(15)


def test_gradient_monitor_outputs_finite_values_and_leaves_no_grads():
    torch.manual_seed(0)
    model = ToyMATDHead()
    x = torch.randn(6, 3)
    loss_dicts = _loss_dicts(model, x)

    monitor = GradientConflictMonitor()
    diag = monitor.compute(loss_dicts, model)

    assert diag["grad_probe_param_count"] > 0
    assert diag["grad_probe_group_count"] == 4.0
    assert diag["grad_group_diffusion_bridge_norm"] > 0.0
    assert math.isfinite(diag["grad_cos_diffusion_bridge__patch_field"])
    assert all(param.grad is None for param in model.parameters())

    total = sum(group[next(iter(group))] for group in loss_dicts.values())
    total.backward()
    assert any(param.grad is not None for param in model.parameters() if param.requires_grad)


def test_gradient_monitor_handles_missing_and_zero_grad_groups():
    torch.manual_seed(0)
    model = ToyMATDHead()
    x = torch.randn(4, 3)
    y, _h = model(x)
    loss_dicts = {
        "diffusion_bridge": {"loss_diffusion_bridge": torch.mean(y * 0.0)},
        "patch_field": {},
    }

    diag = GradientConflictMonitor().compute(loss_dicts, model, components=["encoder"])

    assert diag["grad_group_diffusion_bridge_norm"] == 0.0
    assert diag["grad_group_patch_field_norm"] == 0.0
    assert diag["grad_group_mechanism_norm"] == 0.0
    assert all(param.grad is None for param in model.parameters())
