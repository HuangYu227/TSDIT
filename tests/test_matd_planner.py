import torch

from mmldm.matd.dit import T2PDenoiser
from mmldm.matd.losses import PlanConsistencyLoss
from mmldm.matd.matd_model import MATDConfig
from mmldm.matd.planner import TextLatentPlanGenerator, build_canonical_meta


def test_canonical_meta_invariants():
    meta = build_canonical_meta(batch_size=3, n_patches=5, device=torch.device("cpu"))

    assert meta.shape == (3, 5, 9)
    assert torch.allclose(meta[..., 3].sum(dim=-1), torch.ones(3))
    assert torch.allclose(meta[..., 5].sum(dim=-1), torch.ones(3))
    assert torch.all(meta[..., 0] <= meta[..., 1])
    assert torch.all(meta[:, 1:, 0] >= meta[:, :-1, 0])
    assert torch.allclose(meta[..., 6], meta[..., 5] / meta[..., 3].clamp_min(1e-8))


def test_text_latent_plan_generator_shapes_and_stable_meta():
    torch.manual_seed(0)
    planner = TextLatentPlanGenerator(text_dim=16, hidden_dim=16, n_heads=4, n_patches=8, depth=2)
    text_a = torch.randn(2, 6, 16)
    text_b = text_a.clone()
    text_b[:, 0] = text_b[:, 0] + 1.0
    pooled_a = text_a.mean(dim=1)
    pooled_b = text_b.mean(dim=1)

    out_a = planner(text_a, pooled_text=pooled_a, n_patches=4)
    out_b = planner(text_b, pooled_text=pooled_b, n_patches=4)

    assert out_a.plan_tokens.shape == (2, 4, 16)
    assert out_a.plan_global.shape == (2, 16)
    assert out_a.meta.shape == (2, 4, 9)
    assert out_a.plan_mu.shape == (2, 4, 16)
    assert out_a.plan_logvar.shape == (2, 4, 16)
    assert out_a.plan_kl is not None and out_a.plan_kl.item() >= 0.0
    assert torch.allclose(out_a.meta, out_b.meta)
    assert not torch.allclose(out_a.plan_tokens, out_b.plan_tokens)


def test_plan_consistency_loss_outputs_finite_terms():
    loss_fn = PlanConsistencyLoss(plan_dim=16, text_dim=16)
    plan_global = torch.randn(4, 16)
    pooled = torch.randn(4, 16)
    x0 = torch.rand(4, 12, 1)

    losses = loss_fn(plan_global, pooled, x0, plan_kl=torch.tensor(0.5))

    assert losses["loss_plan_total"].item() >= 0.0
    assert torch.isfinite(losses["loss_plan_total"])
    assert torch.isfinite(losses["loss_plan_stats"])
    assert torch.isfinite(losses["loss_plan_contrast"])
    assert losses["loss_plan_kl"].item() == 0.5


def test_denoiser_accepts_plan_tokens_without_changing_output_shape():
    denoiser = T2PDenoiser(input_dim=16, output_dim=16, text_dim=12, hidden_dim=16, n_heads=4, n_layers=1)
    z = torch.randn(2, 5, 16)
    t = torch.tensor([1, 2])
    text = torch.randn(2, 7, 12)
    pooled = torch.randn(2, 12)
    meta = build_canonical_meta(2, 5, z.device, z.dtype)
    plan_tokens = torch.randn(2, 5, 16)
    plan_global = torch.randn(2, 16)

    out = denoiser(
        z,
        t,
        text,
        meta,
        pooled,
        plan_tokens=plan_tokens,
        plan_global=plan_global,
    )

    assert out.shape == z.shape


def test_matd_config_accepts_latent_planner_knobs():
    cfg = MATDConfig(
        planner_mode="latent",
        planner_depth=4,
        layout_source="canonical",
        lambda_plan_stats=0.3,
        lambda_plan_contrast=0.07,
        lambda_plan_kl=0.002,
        lambda_plan_layout=0.0,
        plan_residual_scale=0.15,
        plan_context_scale=0.8,
    )

    assert cfg.planner_mode == "latent"
    assert cfg.planner_depth == 4
    assert cfg.layout_source == "canonical"
    assert cfg.lambda_plan_stats == 0.3
    assert cfg.lambda_plan_kl == 0.002
    assert cfg.plan_context_scale == 0.8
