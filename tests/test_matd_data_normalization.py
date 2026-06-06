import pandas as pd
import torch

from mmldm.matd.data_adapter import MATDDataset


def _write_tiny_csv(root):
    rows = []
    for i, values in enumerate(
        [
            [0.0, 1.0, 2.0, 3.0],
            [10.0, 11.0, 12.0, 13.0],
            [20.0, 21.0, 22.0, 23.0],
            [30.0, 31.0, 32.0, 33.0],
            [40.0, 41.0, 42.0, 43.0],
            [50.0, 51.0, 52.0, 53.0],
        ]
    ):
        rows.append(
            {
                "SampleID": i,
                "SampleNumID": i,
                "TimeInterval": 4,
                "Text": f"series {i}",
                "TextEmbedding": "[0 0]",
                "OT": str(values),
            }
        )
    pd.DataFrame(rows).to_csv(root / "embedding_cleaned_tiny_4.csv", index=False)


def test_per_sample_normalization_preserves_legacy_bounds(tmp_path):
    _write_tiny_csv(tmp_path)
    ds = MATDDataset(
        data_dir=str(tmp_path),
        split="train",
        dataset_type="csv",
        datasets=["tiny"],
        time_interval=4,
        split_ratio=(1.0, 0.0, 0.0),
        normalization="per_sample",
    )

    x, _text, ts_min, ts_max = ds[0]

    assert torch.allclose(x, torch.tensor([0.0, 1 / 3, 2 / 3, 1.0]))
    assert ts_max - ts_min == 3.0


def test_global_minmax_normalization_uses_dataset_bounds(tmp_path):
    _write_tiny_csv(tmp_path)
    ds = MATDDataset(
        data_dir=str(tmp_path),
        split="train",
        dataset_type="csv",
        datasets=["tiny"],
        time_interval=4,
        split_ratio=(1.0, 0.0, 0.0),
        normalization="global_minmax",
    )

    x, _text, ts_min, ts_max = ds[0]

    assert ts_min == 0.0
    assert ts_max == 53.0
    raw = x * (ts_max - ts_min) + ts_min
    assert torch.allclose(raw[1:] - raw[:-1], torch.ones(3), atol=1e-5)
    assert float(raw.min().item()) in {0.0, 10.0, 20.0, 30.0, 40.0, 50.0}
