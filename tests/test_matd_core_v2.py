import torch
import torch.nn as nn

import mmldm.matd.matd_model as matd_model
from mmldm.matd.matd_model import MATDConfig, MATDModel


class _FakeTextEncoder(nn.Module):
    def __init__(self, *args, model_dim=16, **kwargs):
        super().__init__()
        self.dim = model_dim
        self.anchor = nn.Parameter(torch.zeros(1))

    def forward(self, texts):
        b = len(texts)
        device = self.anchor.device
        base = torch.arange(b * 5 * self.dim, device=device, dtype=torch.float32)
        token_hidden = base.view(b, 5, self.dim) / 1000.0
        pooled = token_hidden.mean(dim=1)
        attention_mask = torch.ones(b, 5, device=device, dtype=torch.long)
        return token_hidden, pooled, attention_mask


def _tiny_cfg(**overrides):
    values = dict(
        architecture="core_v2",
        embed_dim=16,
        model_dim=16,
        text_dim=16,
        text_frozen=True,
        min_tokens=2,
        max_tokens=4,
        target_tokens=4,
        target_seg_len=4,
        ref_len=4,
        min_len=2,
        max_len=8,
        rfft_win=8,
        encoder_context_depth=1,
        encoder_num_heads=4,
        planner_heads=4,
        planner_depth=1,
        scci_heads=4,
        n_slots=2,
        slot_iters=1,
        n_mech=2,
        n_segments=2,
        max_lag=1,
        n_experts=2,
        top_k=1,
        dit_dim=16,
        dit_heads=4,
        dit_depth=1,
        decoder_hidden=16,
        decoder_context_depth=1,
        decoder_context_heads=4,
        decoder_chunk_size=4,
        decoder_field_blocks=1,
        decoder_output_activation="none",
        decoder_latent_noise_std=0.0,
        timesteps=10,
        ddim_steps=2,
        lambda_plan_stats=0.0,
        lambda_plan_contrast=1.0,
        lambda_plan_kl=0.001,
        lambda_plan=0.05,
        lambda_causal=0.0,
        lambda_moe=0.0,
        lambda_scci=0.0,
        lambda_endpoint_moment=0.0,
        lambda_endpoint_delta=0.0,
        lambda_endpoint_acf=0.0,
        use_causal_guidance_in_sampling=False,
    )
    values.update(overrides)
    return MATDConfig(**values)


def test_core_v2_forward_train_skips_fragmented_modules(monkeypatch):
    monkeypatch.setattr(matd_model, "MATDTextEncoder", _FakeTextEncoder)
    torch.manual_seed(0)
    model = MATDModel(_tiny_cfg())
    x0 = torch.rand(2, 12)

    out = model.forward_train(x0, ["cold air", "warm air"], stage=0)

    terms = out["loss_terms"]
    assert torch.isfinite(out["loss_total"])
    assert "loss_diffusion_bridge" in terms
    assert "loss_patch_field" in terms
    assert "loss_plan_total" in terms
    assert out["router_p"] is None
    assert out["prior_p"] is None
    assert out["A0"] is None
    assert out["Alags"] is None
    assert out["meta"].shape == (2, out["z0"].shape[1], 9)
    assert torch.allclose(out["meta"][..., 3].sum(dim=-1), torch.ones(2))
    assert "mechanism" not in out["loss_dicts"]


def test_core_v2_generate_does_not_call_causal(monkeypatch):
    monkeypatch.setattr(matd_model, "MATDTextEncoder", _FakeTextEncoder)

    class _FailingCausal(nn.Module):
        def forward(self, *args, **kwargs):
            raise AssertionError("core_v2 generation should not call causal guidance")

    torch.manual_seed(1)
    model = MATDModel(_tiny_cfg())
    model.causal = _FailingCausal()
    gen = model.generate(["a", "b"], target_length=12, ddim_steps=2, cfg_scale=1.0)

    assert gen.shape == (2, 12, 1)
    assert torch.isfinite(gen).all()


def test_matd_config_core_v2_defaults():
    cfg = MATDConfig()
    assert cfg.architecture == "core_v2"
    assert cfg.layout_source == "canonical"
    assert cfg.use_oracle_meta_prob == 0.0
    assert cfg.use_causal_guidance_in_sampling is False
