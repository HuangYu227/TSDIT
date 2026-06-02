"""Data adapter bridging existing TIGER datasets with the MATD framework.

Provides :class:`MATDDataset` which wraps the same data sources used by
:class:`mmldm.tiger.data.dataset.TIGERDataset` (CSV and weather_npy formats)
but returns simplified ``(x0_tensor, text_str)`` pairs consumed by the MATD
training loop.  Time series are normalised per-sample to [0, 1].

Also provides :class:`MATDDataModule` which assembles train / val / test
DataLoaders with a padding collate function and configurable batch size.

Example::

    from mmldm.matd.data_adapter import MATDDataModule

    dm = MATDDataModule(data_dir="data/weather", batch_size=32)
    for x0, texts in dm.train_dataloader():
        # x0: (B, T) float tensor in [0, 1]
        # texts: list[str] of length B
        ...
"""

from __future__ import annotations

import ast
import os
import random
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ===========================================================================
#  1. MATD Dataset
# ===========================================================================


class MATDDataset(Dataset):
    """Time-series dataset returning (x0, text) pairs for MATD training.

    Supports two data formats inherited from TIGER:

    1. **weather_npy** -- VerbalTS-style .npy files:
       ``{split}_ts.npy`` and ``{split}_text_caps.npy``.
    2. **csv** -- T2S-style CSV files:
       ``embedding_cleaned_{dataset}_{time_interval}.csv`` with columns
       ``SampleID, Text, OT, ...``.

    Each sample is normalised per-sample to [0, 1] using min-max scaling.
    A random caption is chosen when multiple captions are available.

    Args:
        data_dir: Root directory containing the data files.
        split: One of ``"train"``, ``"val"``, or ``"test"``.
        dataset_type: ``"weather_npy"`` or ``"csv"``.
        datasets: List of dataset names for CSV mode (e.g. ``["ETTh1"]``).
        time_interval: Series length for CSV file lookup (24, 48, 96).
        max_samples: Optional cap on the number of samples loaded.
        seed: Random seed for reproducible train/val/test splits (CSV mode).
        split_ratio: ``(train_frac, val_frac, test_frac)`` for random splits.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        dataset_type: str = "weather_npy",
        datasets: Optional[list[str]] = None,
        time_interval: int = 24,
        max_samples: Optional[int] = None,
        seed: int = 123,
        split_ratio: tuple[float, float, float] = (0.8, 0.1, 0.1),
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.dataset_type = dataset_type

        if dataset_type == "weather_npy":
            self._load_weather_npy()
        elif dataset_type == "csv":
            self._load_csv(datasets, time_interval, seed, split_ratio)
        else:
            raise ValueError(f"Unknown dataset_type: {dataset_type!r}")

        if max_samples is not None:
            n = min(len(self.ts_data), max_samples)
            self.ts_data = self.ts_data[:n]
            self.caps = self.caps[:n]

        # Per-sample min-max normalisation to [0, 1]
        ts_min = self.ts_data.min(axis=1, keepdims=True)
        ts_max = self.ts_data.max(axis=1, keepdims=True)
        ts_range = np.clip(ts_max - ts_min, a_min=1e-8, a_max=None)
        self.ts_norm = (self.ts_data - ts_min) / ts_range  # (N, T)

        # Store original-scale bounds for downstream de-normalisation
        self.ts_min = ts_min.squeeze(-1).astype(np.float32)
        self.ts_max = ts_max.squeeze(-1).astype(np.float32)

    # ------------------------------------------------------------------
    #  Data loading helpers
    # ------------------------------------------------------------------

    def _load_weather_npy(self) -> None:
        """Load VerbalTS-style .npy weather data."""
        ts_path = os.path.join(self.data_dir, f"{self.split}_ts.npy")
        caps_path = os.path.join(self.data_dir, f"{self.split}_text_caps.npy")

        self.ts_data = np.load(ts_path).astype(np.float32)  # (N, T)
        raw_caps = np.load(caps_path, allow_pickle=True)  # (N, n_caps)

        # Flatten to list of lists for uniform access
        self.caps: list[list[str]] = [
            list(row) if hasattr(row, "__iter__") else [str(row)]
            for row in raw_caps
        ]

    def _load_csv(
        self,
        datasets: Optional[list[str]],
        time_interval: int,
        seed: int,
        split_ratio: tuple[float, float, float],
    ) -> None:
        """Load T2S-style CSV data."""
        import pandas as pd

        if datasets is None:
            datasets = ["ETTh1"]

        frames = []
        for ds in datasets:
            fpath = os.path.join(
                self.data_dir, f"embedding_cleaned_{ds}_{time_interval}.csv"
            )
            if os.path.exists(fpath):
                frames.append(pd.read_csv(fpath))

        if not frames:
            raise FileNotFoundError(
                f"No CSV files matching 'embedding_cleaned_*_{time_interval}.csv' "
                f"found in {self.data_dir}"
            )

        df = pd.concat(frames, ignore_index=True)

        # Parse time series from 'OT' column (Python list string)
        parsed = [
            ast.literal_eval(item) if isinstance(item, str) else item
            for item in df["OT"]
        ]
        ts_data = np.array(parsed, dtype=np.float32)  # (N, T)

        # Text captions: wrap each in a list for uniform access
        caps: list[list[str]] = [[str(t)] for t in df["Text"].tolist()]

        # Split: use 'split' column if available, otherwise random
        if "split" in df.columns:
            split_map = {"train": "train", "val": "val", "test": "test"}
            mask = df["split"].map(split_map).fillna("train") == self.split
            idx = np.where(mask.values)[0]
        else:
            n = len(ts_data)
            rng = np.random.RandomState(seed)
            perm = rng.permutation(n)
            r_train, r_val, _r_test = split_ratio
            n_train = int(n * r_train)
            n_val = int(n * r_val)
            if self.split == "train":
                idx = perm[:n_train]
            elif self.split in ("val", "valid"):
                idx = perm[n_train : n_train + n_val]
            else:
                idx = perm[n_train + n_val :]

        self.ts_data = ts_data[idx]
        self.caps = [caps[i] for i in idx]

    # ------------------------------------------------------------------
    #  Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.ts_data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        """Return a single sample.

        Args:
            idx: Sample index.

        Returns:
            ``(x0, text)`` where ``x0`` is a float32 tensor of shape (T,)
            normalised to [0, 1], and ``text`` is a caption string.
        """
        x0 = torch.from_numpy(self.ts_norm[idx])  # (T,)

        # Random caption selection when multiple are available
        caps = self.caps[idx]
        if isinstance(caps, (list, np.ndarray)):
            text = str(caps[random.randint(0, len(caps) - 1)])
        else:
            text = str(caps)

        return x0, text


# ===========================================================================
#  2. Collate function (pad to batch max length)
# ===========================================================================


def matd_collate_fn(
    batch: list[tuple[torch.Tensor, str]],
) -> tuple[torch.Tensor, list[str]]:
    """Collate (x0, text) pairs by padding sequences to the batch max length.

    All time series in a batch are right-padded with zeros to match the
    longest sequence.  This is a no-op when all series share the same
    length (the common case for fixed-interval datasets).

    Args:
        batch: List of ``(x0_tensor (T_i,), text_str)`` tuples.

    Returns:
        ``(x0_padded, texts)`` where ``x0_padded`` is (B, T_max) and
        ``texts`` is a list of B caption strings.
    """
    xs, texts = zip(*batch)

    # All same length -- fast path (no padding needed)
    lengths = {x.shape[0] for x in xs}
    if len(lengths) == 1:
        return torch.stack(xs), list(texts)

    # Pad to max length in this batch
    max_len = max(lengths)
    padded = []
    for x in xs:
        if x.shape[0] < max_len:
            pad = torch.zeros(max_len - x.shape[0], dtype=x.dtype)
            x = torch.cat([x, pad], dim=0)
        padded.append(x)

    return torch.stack(padded), list(texts)


# ===========================================================================
#  3. DataModule (train / val / test DataLoaders)
# ===========================================================================


class MATDDataModule:
    """Assembles train, validation, and test DataLoaders for MATD.

    Creates three :class:`MATDDataset` instances (one per split) and wraps
    each in a :class:`~torch.utils.data.DataLoader` with the
    :func:`matd_collate_fn` padding collate.

    Args:
        data_dir: Root directory containing the data files.
        dataset_type: ``"weather_npy"`` or ``"csv"``.
        datasets: List of dataset names for CSV mode.
        time_interval: Series length for CSV file lookup.
        batch_size: Per-DataLoader mini-batch size.
        num_workers: Number of DataLoader worker processes.
        max_samples: Optional cap applied to each split independently.
        seed: Random seed for reproducible splits.
        split_ratio: ``(train, val, test)`` fraction tuple.
        pin_memory: Whether to use pinned memory in DataLoaders.

    Attributes:
        train_ds: Training split dataset.
        val_ds: Validation split dataset.
        test_ds: Test split dataset.

    Example::

        dm = MATDDataModule(data_dir="data/weather", batch_size=64)
        train_loader = dm.train_dataloader()
        for x0, texts in train_loader:
            assert x0.ndim == 2  # (B, T)
            assert len(texts) == x0.shape[0]
    """

    def __init__(
        self,
        data_dir: str,
        dataset_type: str = "weather_npy",
        datasets: Optional[list[str]] = None,
        time_interval: int = 24,
        batch_size: int = 32,
        num_workers: int = 0,
        max_samples: Optional[int] = None,
        seed: int = 123,
        split_ratio: tuple[float, float, float] = (0.8, 0.1, 0.1),
        pin_memory: bool = True,
    ) -> None:
        self.data_dir = data_dir
        self.dataset_type = dataset_type
        self.datasets = datasets
        self.time_interval = time_interval
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_samples = max_samples
        self.seed = seed
        self.split_ratio = split_ratio
        self.pin_memory = pin_memory

        # Build datasets eagerly so errors surface at construction time
        common = dict(
            data_dir=data_dir,
            dataset_type=dataset_type,
            datasets=datasets,
            time_interval=time_interval,
            max_samples=max_samples,
            seed=seed,
            split_ratio=split_ratio,
        )
        self.train_ds = MATDDataset(split="train", **common)
        self.val_ds = MATDDataset(split="val", **common)
        self.test_ds = MATDDataset(split="test", **common)

    # ------------------------------------------------------------------
    #  DataLoader factories
    # ------------------------------------------------------------------

    def _make_loader(self, dataset: MATDDataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=matd_collate_fn,
            pin_memory=self.pin_memory,
            drop_last=False,
        )

    def train_dataloader(self) -> DataLoader:
        """Return the training DataLoader (shuffled)."""
        return self._make_loader(self.train_ds, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """Return the validation DataLoader (not shuffled)."""
        return self._make_loader(self.val_ds, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        """Return the test DataLoader (not shuffled)."""
        return self._make_loader(self.test_ds, shuffle=False)

    # ------------------------------------------------------------------
    #  Convenience properties
    # ------------------------------------------------------------------

    @property
    def seq_len(self) -> int:
        """Sequence length T (assumes all splits share the same length)."""
        return int(self.train_ds.ts_data.shape[1])

    def __repr__(self) -> str:
        return (
            f"MATDDataModule(data_dir={self.data_dir!r}, "
            f"type={self.dataset_type!r}, "
            f"train={len(self.train_ds)}, val={len(self.val_ds)}, "
            f"test={len(self.test_ds)}, batch_size={self.batch_size})"
        )
