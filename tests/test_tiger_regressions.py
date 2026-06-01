from argparse import Namespace

import torch

from mmldm.tiger.data.dataset import TIGERCollateFn
from mmldm.tiger.dit_model import TIGERDiT
from mmldm.tiger.image_to_ts import RowRasterDecoder
from mmldm.tiger.train import (
    apply_cli_overrides,
    denormalize_ts_batch,
    get_default_config,
)
from mmldm.tiger.ts_to_image import RowRasterEncoder


def _cli_args(**overrides):
    values = {
        "data_dir": "/tmp/data",
        "save_dir": None,
        "log_dir": None,
        "model_path": None,
        "epochs": None,
        "batch_size": None,
        "lr": None,
        "warmup_steps": None,
        "seed": None,
        "val_interval": None,
        "display_interval": None,
        "save_interval": None,
        "dataset_type": None,
        "datasets": None,
        "time_interval": None,
        "image_size": None,
        "n_fft": None,
        "hop_length": None,
        "num_steps": None,
        "channels": None,
        "nheads": None,
        "layers": None,
        "n_var": None,
        "multipatch_num": None,
        "eval_only": False,
    }
    values.update(overrides)
    return Namespace(**values)


def test_cli_does_not_override_json_with_parser_defaults():
    cfg = get_default_config()
    cfg["batch_size"] = 512
    cfg["diffusion"]["n_var"] = 1
    cfg["diffusion"]["multipatch_num"] = 1
    cfg["condition"]["cfg_dropout"] = 0.5

    out = apply_cli_overrides(cfg, _cli_args())

    assert out["data_dir"] == "/tmp/data"
    assert out["batch_size"] == 512
    assert out["diffusion"]["n_var"] == 1
    assert out["diffusion"]["multipatch_num"] == 1
    assert out["condition"]["cfg_dropout"] == 0.5


def test_cli_sets_eval_only_without_touching_checkpoint_path():
    cfg = get_default_config()
    cfg["model_path"] = "ckpts/best.pth"

    out = apply_cli_overrides(cfg, _cli_args(eval_only=True))

    assert out["eval_only"] is True
    assert out["model_path"] == "ckpts/best.pth"


def test_denormalize_ts_batch_keeps_batch_time_shape():
    ts_norm = torch.full((2, 3), 0.5)
    ts_min = torch.tensor([[1.0], [10.0]])
    ts_max = torch.tensor([[3.0], [14.0]])

    out = denormalize_ts_batch(ts_norm, ts_min, ts_max)

    assert out.shape == (2, 3)
    assert torch.allclose(out[0], torch.tensor([2.0, 2.0, 2.0]))
    assert torch.allclose(out[1], torch.tensor([12.0, 12.0, 12.0]))


def test_tiger_collate_squeezes_scalar_norm_values():
    batch = [
        {
            "image": torch.zeros(3, 8, 8),
            "ts": torch.zeros(4),
            "ts_min": torch.tensor([0.0]),
            "ts_max": torch.tensor([1.0]),
            "cap": "sample",
            "tp": torch.arange(4),
            "ts_len": 4,
        },
        {
            "image": torch.ones(3, 8, 8),
            "ts": torch.ones(4),
            "ts_min": torch.tensor([2.0]),
            "ts_max": torch.tensor([4.0]),
            "cap": "sample",
            "tp": torch.arange(4),
            "ts_len": 4,
        },
    ]

    out = TIGERCollateFn()(batch)

    assert out["ts_min"].shape == (2,)
    assert out["ts_max"].shape == (2,)


def test_dit_accepts_multi_anchor_condition_map():
    cfg = {
        "num_steps": 10,
        "channels": 16,
        "nheads": 4,
        "layers": 1,
        "diffusion_embedding_dim": 16,
        "base_patch": 4,
        "patch_scale": 2,
        "multipatch_num": 3,
        "in_channels": 3,
        "condition_type": "adaLN",
        "attention_mask_type": "parallel",
    }
    model = TIGERDiT(cfg)
    image = torch.randn(2, 3, 32, 32)
    diffusion_step = torch.randint(0, cfg["num_steps"], (2,))
    attr_emb = torch.randn(2, cfg["channels"], 8, cfg["multipatch_num"])

    out = model(image, diffusion_step, attr_emb)

    assert out.shape == image.shape


def test_row_raster_is_single_channel_row_major_roundtrip():
    ts = torch.tensor([
        [2.0, 4.0, 6.0, 8.0, 10.0],
        [-1.0, 0.0, 1.0, 2.0, 3.0],
    ])
    encoder = RowRasterEncoder(patch_size=4)

    image, norm_params = encoder.encode(ts)

    assert image.shape == (2, 1, 4, 4)
    flat = image[:, 0].reshape(2, -1)
    expected_norm = (ts - ts.amin(dim=-1, keepdim=True)) / (
        ts.amax(dim=-1, keepdim=True) - ts.amin(dim=-1, keepdim=True)
    )
    assert torch.allclose(flat[:, : ts.shape[1]], expected_norm)
    assert torch.allclose(flat[:, ts.shape[1]:], torch.full((2, 11), 0.5))

    decoded = RowRasterDecoder().decode(image, ts_length=ts.shape[1], norm_params=norm_params)

    assert decoded.shape == ts.shape
    assert torch.allclose(decoded, ts)
