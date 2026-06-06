import torch

from mmldm.matd.losses import EndpointSeriesLoss
from mmldm.matd.matd_model import MATDConfig


def test_endpoint_series_loss_is_zero_for_perfect_prediction():
    target = torch.rand(3, 12, 1)
    loss_fn = EndpointSeriesLoss()

    losses = loss_fn(target.clone(), target)

    assert losses["loss_endpoint_total"].item() < 1e-7
    assert losses["loss_endpoint_recon"].item() < 1e-7
    assert losses["loss_endpoint_delta"].item() < 1e-7


def test_endpoint_series_loss_penalizes_moment_and_structure_errors():
    target = torch.linspace(0.0, 1.0, 16).view(1, 16, 1).repeat(2, 1, 1)
    pred = torch.full_like(target, 0.5)
    loss_fn = EndpointSeriesLoss(
        recon_weight=1.0,
        moment_weight=0.2,
        delta_weight=0.2,
        acf_weight=0.1,
    )

    losses = loss_fn(pred, target)

    assert losses["loss_endpoint_total"].item() > 0.0
    assert losses["loss_endpoint_moment"].item() > 0.0
    assert losses["loss_endpoint_delta"].item() > 0.0


def test_matd_config_accepts_endpoint_training_knobs():
    cfg = MATDConfig(
        lambda_endpoint_recon=0.7,
        lambda_endpoint_moment=0.3,
        lambda_endpoint_delta=0.2,
        lambda_endpoint_acf=0.1,
        decoder_latent_noise_std=0.01,
        endpoint_latent_clip=10.0,
    )

    assert cfg.lambda_endpoint_recon == 0.7
    assert cfg.decoder_latent_noise_std == 0.01
    assert cfg.endpoint_latent_clip == 10.0
