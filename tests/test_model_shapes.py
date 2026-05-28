import torch

from mmldm.configuration_mmldm import MMLDMDiTConfig, MMLDMVAEConfig
from mmldm.modeling_mmldm_dit import MMLDMDiTModel
from mmldm.modeling_mmldm_vae import MMLDMVAEModel


def test_tri_band_vae_poe_outputs_latent_dim():
    cfg = MMLDMVAEConfig(
        ts_channels=2,
        dim=16,
        latent_dim=6,
        text_dim=8,
        num_conv_layers=1,
        decoder_num_blocks=1,
        use_tri_band=True,
        text_num_tokens=4,
    )
    model = MMLDMVAEModel(cfg)
    output = model([torch.randn(12, 2), torch.randn(10, 2)])

    assert output["latents"][0].shape == (12, cfg.latent_dim)
    assert output["latents"][1].shape == (10, cfg.latent_dim)
    assert output["recon"].shape == (1, 22, cfg.ts_channels)


def test_text_encoder_returns_multiple_condition_tokens():
    cfg = MMLDMVAEConfig(
        ts_channels=1,
        dim=16,
        latent_dim=8,
        text_dim=10,
        text_num_tokens=6,
    )
    model = MMLDMVAEModel(cfg)
    tokens, film_params = model.encode_text_tokens(torch.randn(3, cfg.text_dim))

    assert tokens.shape == (3, cfg.text_num_tokens, cfg.latent_dim)
    assert len(film_params) == 2


def test_dit_accepts_multi_token_text_condition():
    cfg = MMLDMDiTConfig(
        ts_in_channels=8,
        ts_out_channels=8,
        text_in_channels=8,
        text_out_channels=8,
        txt_dim=16,
        emb_dim=16,
        heads=2,
        head_dim=8,
        num_layers=2,
        text_latent_dim=10,
    )
    model = MMLDMDiTModel(cfg)
    ts_shape = torch.tensor([[12], [10]])
    text_shape = torch.tensor([[6], [6]])
    output = model(
        ts=torch.randn(22, 8),
        text=torch.randn(12, 8),
        ts_shape=ts_shape,
        text_shape=text_shape,
        timestep=torch.rand(22),
        text_latent=torch.randn(2, 10),
    )

    assert output.ts_sample.shape == (22, 8)
