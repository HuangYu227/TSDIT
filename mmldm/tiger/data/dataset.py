import os
import json
import ast
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from ..ts_to_image import TSToImageEncoder


class TIGERDataset(Dataset):
    """Dataset for TIGER: Text+Image Guided TS Generation.
    
    Supports two data formats:
    1. Weather .npy format (VerbalTS style): train_ts.npy, train_text_caps.npy, etc.
    2. CSV format (T2S style): embedding_cleaned_{dataset}_{length}.csv
    """

    def __init__(self, data_dir, split="train", dataset_type="weather_npy",
                 datasets=None, time_interval=24, max_samples=None,
                 image_size=64, n_fft=64, hop_length=8, epsilon_quantile=0.1,
                 seed=123, split_ratio=(0.8, 0.1, 0.1),
                 ref_image_mode: str | None = None,
                 ref_image_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.dataset_type = dataset_type
        self.image_size = image_size
        self.max_samples = max_samples
        self.seed = seed
        self.split_ratio = split_ratio  # (train, val, test) for CSV without 'split' column
        self.ref_image_mode = ref_image_mode
        self.ref_image_size = ref_image_size

        # TS→Image encoder
        self.ts_to_image = TSToImageEncoder(
            image_size=image_size, n_fft=n_fft, hop_length=hop_length,
            epsilon_quantile=epsilon_quantile
        )

        if dataset_type == "weather_npy":
            self._load_weather_npy()
        elif dataset_type == "csv":
            self._load_csv(datasets, time_interval)
        else:
            raise ValueError(f"Unknown dataset_type: {dataset_type}")

        if max_samples is not None:
            self.n_samples = min(self.n_samples, max_samples)

        # Pre-compute images for all samples
        self._precompute_images()

        # Pre-compute reference images (for multi-modal conditioning)
        self._precompute_ref_images()

    def _load_weather_npy(self):
        """Load Weather dataset in VerbalTS .npy format."""
        ts_path = os.path.join(self.data_dir, f"{self.split}_ts.npy")
        caps_path = os.path.join(self.data_dir, f"{self.split}_text_caps.npy")
        attrs_path = os.path.join(self.data_dir, f"{self.split}_attrs_idx.npy")

        self.ts_data = np.load(ts_path)  # (n_samples, n_steps)
        self.caps = np.load(caps_path, allow_pickle=True)  # (n_samples, n_caps)
        self.attrs = np.load(attrs_path) if os.path.exists(attrs_path) else None

        self.n_samples = self.ts_data.shape[0]
        self.n_steps = self.ts_data.shape[1]
        self.time_points = np.arange(self.n_steps)

        # Global min/max for T2S metric normalization on [0,1] scale.
        # Must be present for _compute_gen_metrics (same contract as _load_csv).
        self.global_ts_min = float(np.min(self.ts_data))
        self.global_ts_max = float(np.max(self.ts_data))

    def _load_csv(self, datasets, time_interval):
        """Load T2S-style CSV dataset.

        Expected file: ``embedding_cleaned_{dataset}_{time_interval}.csv``
        Columns: SampleID, SampleNumID, TimeInterval, Text, TextEmbedding, OT

        Args:
            datasets: list of dataset names (e.g. ["ETTh1"]).
            time_interval: series length (24, 48, or 96).
        """
        import pandas as pd

        if datasets is None:
            datasets = ["ETTh1"]

        all_dfs = []
        for ds in datasets:
            fname = f"embedding_cleaned_{ds}_{time_interval}.csv"
            fpath = os.path.join(self.data_dir, fname)
            if os.path.exists(fpath):
                df = pd.read_csv(fpath)
                all_dfs.append(df)

        if not all_dfs:
            raise FileNotFoundError(
                f"No CSV files matching 'embedding_cleaned_*_{time_interval}.csv' "
                f"found in {self.data_dir}"
            )

        df = pd.concat(all_dfs, ignore_index=True)

        # Parse time series from 'OT' column (Python list string)
        parsed = [
            ast.literal_eval(item) if isinstance(item, str) else item
            for item in df["OT"]
        ]
        ts_data = np.array(parsed, dtype=np.float32)  # (N, T)

        # Text captions
        caps = [[t] for t in df["Text"].tolist()]  # wrap in list for consistency

        # Split: use 'split' column if available (CaTSG), otherwise random split.
        # The random split uses split_ratio (default 80/10/10) for a proper
        # three-way partition, matching the project's split_dataset.py convention.
        if "split" in df.columns:
            split_map = {"train": "train", "val": "val", "test": "test"}
            mask = df["split"].map(split_map).fillna("train") == self.split
            idx = np.where(mask.values)[0]
        else:
            n = len(ts_data)
            rng = np.random.RandomState(self.seed)
            perm = rng.permutation(n)
            r_train, r_val, r_test = self.split_ratio
            n_train = int(n * r_train)
            n_val  = int(n * r_val)
            # test gets the remainder (avoids off-by-one from float rounding)
            if self.split == "train":
                idx = perm[:n_train]
            elif self.split in ("val", "valid"):
                idx = perm[n_train : n_train + n_val]
            else:  # "test"
                idx = perm[n_train + n_val :]

        self.ts_data = ts_data[idx]
        self.caps = np.array([caps[i] for i in idx], dtype=object)
        self.attrs = None

        self.n_samples = len(self.ts_data)
        self.n_steps = self.ts_data.shape[1]
        self.time_points = np.arange(self.n_steps)

        # Global min/max for scale-leakage-free denormalization
        self.global_ts_min = float(np.min(self.ts_data))
        self.global_ts_max = float(np.max(self.ts_data))

    def _precompute_images(self):
        """Pre-compute GAF/STFT/RP images for all samples."""
        print(f"Pre-computing {self.n_samples} images...")
        ts_tensor = torch.tensor(self.ts_data[:self.n_samples], dtype=torch.float32)
        if ts_tensor.ndim == 1:
            ts_tensor = ts_tensor.unsqueeze(0)

        # Pass raw data to encoder; it handles normalization internally
        # and returns correct NormParams with original-scale min/max
        self.images, self.norm_params = self.ts_to_image.encode(ts_tensor)

        # Store per-sample normalized TS for training
        ts_min = self.norm_params.min_val.unsqueeze(-1)
        ts_max = self.norm_params.max_val.unsqueeze(-1)
        ts_range = (ts_max - ts_min).clamp(min=1e-8)
        self.ts_norm = (ts_tensor - ts_min) / ts_range
        self.ts_min = self.norm_params.min_val
        self.ts_max = self.norm_params.max_val
        print(f"Images computed: {self.images.shape}")

    def _precompute_ref_images(self):
        """Pre-compute reference images for multi-modal conditioning.

        ``ref_image_mode=None`` → no reference images (backward compatible).
        ``ref_image_mode="self"`` → ref_image = same as target image
            (self-conditioning; the DiT sees the clean image as condition).
        ``ref_image_mode="different_encoding"`` → encodes the same TS with
            a different image_size (e.g. 32 vs 64), providing a multi-scale
            conditioning signal.
        """
        if self.ref_image_mode is None:
            self.ref_images = None
            return

        if self.ref_image_mode == "self":
            # Reference image = target image (self-conditioning)
            self.ref_images = self.images.clone()
            print(f"Ref images (self): {self.ref_images.shape}")

        elif self.ref_image_mode == "different_encoding":
            ref_encoder = TSToImageEncoder(
                image_size=self.ref_image_size,
                n_fft=self.ts_to_image.n_fft,
                hop_length=self.ts_to_image.hop_length,
                epsilon_quantile=self.ts_to_image.epsilon_quantile,
            )
            ts_tensor = torch.tensor(self.ts_data[:self.n_samples], dtype=torch.float32)
            if ts_tensor.ndim == 1:
                ts_tensor = ts_tensor.unsqueeze(0)
            self.ref_images, _ = ref_encoder.encode(ts_tensor)
            print(f"Ref images (different_encoding, size={self.ref_image_size}): {self.ref_images.shape}")

        else:
            raise ValueError(f"Unknown ref_image_mode: '{self.ref_image_mode}'. "
                             f"Use None, 'self', or 'different_encoding'.")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Get caption (random choice if multiple)
        caps = self.caps[idx]
        if isinstance(caps, (list, np.ndarray)):
            cap_id = random.randint(0, len(caps) - 1)
            cap = caps[cap_id]
        else:
            cap = caps

        sample = {
            "image": self.images[idx],           # (3, H, W)
            "ts": self.ts_norm[idx],              # (T,) normalized
            "ts_min": self.ts_min[idx],           # scalar
            "ts_max": self.ts_max[idx],           # scalar
            "cap": cap,                           # str
            "tp": self.time_points,               # (T,)
            "ts_len": self.n_steps,
        }

        if self.ref_images is not None:
            sample["ref_image"] = self.ref_images[idx]

        if self.attrs is not None:
            sample["attrs"] = self.attrs[idx]

        return sample


class TIGERCollateFn:
    """Collate function for TIGERDataset."""

    def __call__(self, batch):
        images = torch.stack([b["image"] for b in batch])
        ts = torch.stack([b["ts"] for b in batch])
        ts_min = torch.stack([
            torch.as_tensor(b["ts_min"], dtype=torch.float32).reshape(-1)
            for b in batch
        ])
        ts_max = torch.stack([
            torch.as_tensor(b["ts_max"], dtype=torch.float32).reshape(-1)
            for b in batch
        ])
        if ts_min.shape[-1] == 1:
            ts_min = ts_min.squeeze(-1)
            ts_max = ts_max.squeeze(-1)
        caps = [b["cap"] for b in batch]
        tp = torch.stack([torch.as_tensor(b["tp"]) for b in batch])
        ts_len = batch[0]["ts_len"]

        result = {
            "image": images,
            "ts": ts,
            "ts_min": ts_min,
            "ts_max": ts_max,
            "cap": caps,
            "tp": tp,
            "ts_len": ts_len,
        }

        if "ref_image" in batch[0] and batch[0]["ref_image"] is not None:
            result["ref_image"] = torch.stack([b["ref_image"] for b in batch])

        if "attrs" in batch[0]:
            result["attrs"] = torch.stack([torch.tensor(b["attrs"]) for b in batch])

        return result
