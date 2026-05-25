"""Weather dataset adapter for VerbalTS Weather data.

Reads .npy files from VerbalTS Weather dataset and returns dicts compatible
with the MMLDM training/inference pipeline.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


def _encode_captions_sbert(captions: list[str], device: str = "cpu") -> np.ndarray:
    """Encode a list of text captions into 128-dim SBERT embeddings.

    Caches embeddings to ``text_embeddings_128.npy`` in the data directory.
    """
    try:
        from transformers import AutoModel, AutoTokenizer
        sbert_name = "sentence-transformers/all-MiniLM-L6-v2"
        tokenizer = AutoTokenizer.from_pretrained(sbert_name)
        sbert = AutoModel.from_pretrained(sbert_name).to(device)
        sbert.eval()
        embeddings = []
        batch_size = 64
        with torch.no_grad():
            for i in range(0, len(captions), batch_size):
                batch = captions[i:i + batch_size]
                inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
                emb = sbert(**inputs).last_hidden_state.mean(dim=1)  # (B, 384)
                emb = emb[:, :128]  # truncate to 128 dims
                embeddings.append(emb.cpu().numpy())
        del sbert, tokenizer
        return np.concatenate(embeddings, axis=0).astype(np.float32)
    except Exception as e:
        print(f"  WARNING: SBERT encoding failed ({e}), using hash fallback")
        import hashlib
        emb_list = []
        for cap in captions:
            h = hashlib.sha256(cap.encode()).digest()
            vec = np.array([b / 255.0 for b in h], dtype=np.float32)
            vec = np.resize(vec, 128)
            emb_list.append(vec)
        return np.stack(emb_list, axis=0)


def _load_or_encode_captions(weather_data_dir: str) -> np.ndarray:
    """Load cached embeddings or encode all captions and cache."""
    cache_path = os.path.join(weather_data_dir, "text_embeddings_128.npy")
    if os.path.exists(cache_path):
        return np.load(cache_path)

    # Collect all captions from all splits
    all_captions = []
    for split in ["train", "valid", "test"]:
        caps_file = os.path.join(weather_data_dir, f"{split}_text_caps.npy")
        if os.path.exists(caps_file):
            caps = np.load(caps_file, allow_pickle=True)  # (n_samples, n_captions_per_sample)
            for sample_caps in caps:
                all_captions.append(str(sample_caps[0]))  # use first caption per sample

    print(f"  Encoding {len(all_captions)} captions with SBERT...")
    embeddings = _encode_captions_sbert(all_captions)
    np.save(cache_path, embeddings)
    print(f"  Saved cached embeddings to {cache_path}")
    return embeddings


class WeatherDataset(Dataset):
    """Dataset for VerbalTS Weather .npy data.

    Args:
        weather_data_dir: path to the directory containing
            ``train_ts.npy``, ``test_ts.npy``, etc.
        split: ``"train"``, ``"valid"``, or ``"test"``.
        max_samples: maximum number of samples to load (for debugging).
    """

    def __init__(
        self,
        weather_data_dir: str,
        split: str = "train",
        max_samples: Optional[int] = None,
    ):
        super().__init__()
        self.weather_data_dir = weather_data_dir
        self.split = split

        ts_path = os.path.join(weather_data_dir, f"{split}_ts.npy")
        caps_path = os.path.join(weather_data_dir, f"{split}_text_caps.npy")

        self.ts = np.load(ts_path)  # (n_samples, L, C)
        self.caps = np.load(caps_path, allow_pickle=True)  # (n_samples, n_captions)

        if self.ts.ndim == 2:
            self.ts = self.ts[:, :, np.newaxis]  # (n_samples, L) → (n_samples, L, 1)

        self.n_samples = self.ts.shape[0]
        self.seq_len = self.ts.shape[1]
        self.n_vars = self.ts.shape[2]

        if max_samples is not None:
            self.n_samples = min(self.n_samples, max_samples)
            self.ts = self.ts[:self.n_samples]
            self.caps = self.caps[:self.n_samples]

        # Pre-compute or load SBERT embeddings for all captions (shared across splits)
        all_embeddings = _load_or_encode_captions(weather_data_dir)
        # Map to this split's samples
        self.text_embeddings: np.ndarray = all_embeddings[
            self._split_offset() : self._split_offset() + self.n_samples
        ].copy()

        print(f"  WeatherDataset [{split}]: {self.n_samples} samples, "
              f"seq_len={self.seq_len}, n_vars={self.n_vars}")

    def _split_offset(self) -> int:
        """Compute the starting index of this split in the all-embeddings array."""
        offset = 0
        for s in ["train", "valid", "test"]:
            if s == self.split:
                return offset
            caps_file = os.path.join(self.weather_data_dir, f"{s}_text_caps.npy")
            if os.path.exists(caps_file):
                offset += np.load(caps_file, allow_pickle=True).shape[0]
        return offset

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict:
        ot = self.ts[idx].copy()  # (L, C)
        cap_str = str(self.caps[idx][0])

        # Z-normalization per variable independently
        mean = ot.mean(axis=0, keepdims=True)  # (1, C)
        std = ot.std(axis=0, keepdims=True)    # (1, C)
        std = np.where(std < 1e-8, 1.0, std)
        ot_norm = (ot - mean) / std

        return {
            "text_embedding": torch.from_numpy(self.text_embeddings[idx].copy()),
            "ot": torch.from_numpy(ot_norm),                               # (L, C) normalized
            "ot_lengths": torch.tensor(self.seq_len, dtype=torch.long),
            "ot_means": torch.from_numpy(mean.flatten()).float(),          # (C,)
            "ot_stds": torch.from_numpy(std.flatten()).float(),            # (C,)
            "text_str": cap_str,
            "dataset_name": "Weather",
            "time_interval": self.seq_len,
        }


class WeatherCollateFn:
    """Collate function for Weather dataset with multivariate time series.

    Output dict keys match CollateFn from tsfragment_dataset.py for compatibility.
    """

    def __call__(self, batch: list[dict]) -> dict:
        text_embeddings = torch.stack([s["text_embedding"] for s in batch])  # (B, 128)
        ot_lengths = torch.tensor([s["ot"].shape[0] for s in batch], dtype=torch.long)
        L_max = int(ot_lengths.max().item())
        C = batch[0]["ot"].shape[1]

        B = len(batch)
        ot_padded = torch.zeros(B, L_max, C, dtype=torch.float32)
        ot_means = torch.zeros(B, C, dtype=torch.float32)
        ot_stds = torch.ones(B, C, dtype=torch.float32)
        for i, s in enumerate(batch):
            L = s["ot"].shape[0]
            ot_padded[i, :L, :] = s["ot"]
            ot_means[i] = s["ot_means"]
            ot_stds[i] = s["ot_stds"]

        return {
            "text_embedding": text_embeddings,   # (B, 128)
            "ot": ot_padded,                     # (B, L_max, C) normalized
            "ot_lengths": ot_lengths,            # (B,)
            "ot_means": ot_means,                # (B, C) original means
            "ot_stds": ot_stds,                  # (B, C) original stds
            "text_strs": [s["text_str"] for s in batch],
            "dataset_names": [s["dataset_name"] for s in batch],
            "time_intervals": [s["time_interval"] for s in batch],
        }
