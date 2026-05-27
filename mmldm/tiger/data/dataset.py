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
                 image_size=64, n_fft=64, hop_length=8, epsilon_quantile=0.1):
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.dataset_type = dataset_type
        self.image_size = image_size
        self.max_samples = max_samples

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

    def _load_csv(self, datasets, time_interval):
        """Load TSFragment-600K in T2S CSV format."""
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
            raise FileNotFoundError(f"No CSV files found in {self.data_dir}")
        
        df = pd.concat(all_dfs, ignore_index=True)
        
        # Parse time series and text
        self.ts_data = []
        self.caps = []
        for _, row in df.iterrows():
            ts = np.array(ast.literal_eval(row["OT"]), dtype=np.float32)
            self.ts_data.append(ts)
            self.caps.append([row["Text"]])  # Wrap in list for consistency
        
        self.ts_data = np.array(self.ts_data)
        self.caps = np.array(self.caps, dtype=object)
        self.attrs = None
        
        self.n_samples = len(self.ts_data)
        self.n_steps = self.ts_data.shape[1] if self.ts_data.ndim == 2 else self.ts_data.shape[-1]
        self.time_points = np.arange(self.n_steps)

    def _precompute_images(self):
        """Pre-compute GAF/STFT/RP images for all samples."""
        print(f"Pre-computing {self.n_samples} images...")
        ts_tensor = torch.tensor(self.ts_data[:self.n_samples], dtype=torch.float32)
        if ts_tensor.ndim == 1:
            ts_tensor = ts_tensor.unsqueeze(0)
        
        # Normalize to [0,1] per sample
        ts_min = ts_tensor.min(dim=-1, keepdim=True).values
        ts_max = ts_tensor.max(dim=-1, keepdim=True).values
        ts_range = ts_max - ts_min
        ts_range = torch.clamp(ts_range, min=1e-8)
        ts_norm = (ts_tensor - ts_min) / ts_range
        
        self.images, self.norm_params = self.ts_to_image.encode(ts_norm)
        self.ts_norm = ts_norm
        self.ts_min = ts_min
        self.ts_max = ts_max
        print(f"Images computed: {self.images.shape}")

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
        
        if self.attrs is not None:
            sample["attrs"] = self.attrs[idx]
        
        return sample


class TIGERCollateFn:
    """Collate function for TIGERDataset."""

    def __call__(self, batch):
        images = torch.stack([b["image"] for b in batch])
        ts = torch.stack([b["ts"] for b in batch])
        ts_min = torch.tensor([b["ts_min"] for b in batch])
        ts_max = torch.tensor([b["ts_max"] for b in batch])
        caps = [b["cap"] for b in batch]
        tp = torch.stack([torch.tensor(b["tp"]) for b in batch])
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

        if "attrs" in batch[0]:
            result["attrs"] = torch.stack([torch.tensor(b["attrs"]) for b in batch])

        return result
