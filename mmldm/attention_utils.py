# Copyright 2026 MMLDM Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Attention mask utilities for MMLDM.

Multimodal joint mask for the [all_ts_tokens ; all_text_tokens] flat layout.
All masks are additive: 0 for allowed positions, dtype.min for blocked.
"""

from __future__ import annotations

from typing import Sequence

import torch


def create_multimodal_joint_mask(
    ts_shape: torch.LongTensor,
    text_shape: torch.LongTensor,
    block_sizes: Sequence[Sequence[int]],
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Multimodal adaptive joint causal mask for [ts; text] flat layout.

    Constraints:
    - **Sample isolation**: tokens from different samples never attend.
    - **Block-causal on TS**: within TS, token in block *b* can only attend
      to blocks *<= b* (bidirectional within block, causal across blocks).
    - **Fully bidirectional on cross-modal and text**: TS<->Text and
      Text<->Text are fully visible (within same sample).

    Args:
        ts_shape: ``(B, 1)`` per-sample patched TS lengths.
        text_shape: ``(B, 1)`` per-sample patched text lengths.
        block_sizes: nested list ``[[b1, b2, ...], ...]`` with one inner
            list per sample.  ``sum(block_sizes[i]) == ts_shape[i]``.
        dtype: mask dtype (additive, so use large negative for blocked).
        device: target device.

    Returns:
        ``(1, 1, L_total, L_total)`` additive mask where
        ``L_total = sum(ts_shape) + sum(text_shape)``.
    """
    B = ts_shape.shape[0]
    assert len(block_sizes) == B, (
        f"block_sizes samples {len(block_sizes)} != batch size {B}"
    )

    L_ts = int(ts_shape.sum().item())
    L_text = int(text_shape.sum().item())
    L_total = L_ts + L_text

    # 1. TS side: sample_id + block_id
    ts_sample_ids = torch.zeros(L_ts, dtype=torch.long, device=device)
    ts_block_ids = torch.zeros(L_ts, dtype=torch.long, device=device)

    ts_cum = 0
    for b_idx in range(B):
        cur_ts_len = int(ts_shape[b_idx].item())
        if cur_ts_len == 0:
            continue

        sample_blocks = block_sizes[b_idx]
        assert sum(sample_blocks) == cur_ts_len, (
            f"Sample {b_idx}: block sum {sum(sample_blocks)} != ts_shape {cur_ts_len}"
        )

        pos = 0
        for block_id, bs in enumerate(sample_blocks):
            end = min(pos + bs, cur_ts_len)
            ts_block_ids[ts_cum + pos : ts_cum + end] = block_id
            pos = end

        ts_sample_ids[ts_cum : ts_cum + cur_ts_len] = b_idx
        ts_cum += cur_ts_len

    # 2. Text side: sample_id only
    text_sample_ids = torch.zeros(L_text, dtype=torch.long, device=device)
    text_cum = 0
    for b_idx in range(B):
        cur_text_len = int(text_shape[b_idx].item())
        if cur_text_len == 0:
            continue
        text_sample_ids[text_cum : text_cum + cur_text_len] = b_idx
        text_cum += cur_text_len

    # 3. Global physical layout: [ts ; text]
    global_sample_id = torch.cat([ts_sample_ids, text_sample_ids])  # (L_total,)

    # 4. Sample isolation
    same_sample = (
        global_sample_id.unsqueeze(1) == global_sample_id.unsqueeze(0)
    )  # (L_total, L_total)
    allowed = same_sample.clone()

    # 5. Block-causal on TS->TS region
    ts_causal = (
        ts_block_ids.unsqueeze(1) >= ts_block_ids.unsqueeze(0)
    )  # (L_ts, L_ts)
    allowed[:L_ts, :L_ts] = allowed[:L_ts, :L_ts] & ts_causal

    # Cross-modal (TS<->Text), Text<->Text: fully visible within same sample
    # (already handled by same_sample above)

    # 6. Build additive mask
    mask = torch.full(
        (L_total, L_total), torch.finfo(dtype).min, dtype=dtype, device=device,
    )
    mask[allowed] = 0.0

    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, L_total, L_total)


def create_dit_readonly_text_mask(
    ts_shape: torch.LongTensor,
    text_shape: torch.LongTensor,
    block_sizes: Sequence[Sequence[int]],
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Multimodal DiT mask where text tokens are *read-only*.

    Identical to :func:`create_multimodal_joint_mask` except that
    **Text → TS attention is forbidden**.  Text tokens can only attend
    to other text tokens (within the same sample).  This prevents
    multi-layer relay leakage where text absorbs TS information across
    layers and injects it back into TS via cross-attention.

    Constraints:
    - **Sample isolation**: tokens from different samples never attend.
    - **Block-causal on TS**: within TS, token in block *b* can only attend
      to blocks *<= b* (bidirectional within block, causal across blocks).
    - **TS → Text**: TS tokens can attend to all text tokens (same sample).
    - **Text → Text**: fully bidirectional (within same sample).
    - **Text → TS**: **blocked** (read-only text).
    """
    B = ts_shape.shape[0]
    assert len(block_sizes) == B, (
        f"block_sizes samples {len(block_sizes)} != batch size {B}"
    )

    L_ts = int(ts_shape.sum().item())
    L_text = int(text_shape.sum().item())
    L_total = L_ts + L_text

    # 1. TS side: sample_id + block_id
    ts_sample_ids = torch.zeros(L_ts, dtype=torch.long, device=device)
    ts_block_ids = torch.zeros(L_ts, dtype=torch.long, device=device)

    ts_cum = 0
    for b_idx in range(B):
        cur_ts_len = int(ts_shape[b_idx].item())
        if cur_ts_len == 0:
            continue

        sample_blocks = block_sizes[b_idx]
        assert sum(sample_blocks) == cur_ts_len, (
            f"Sample {b_idx}: block sum {sum(sample_blocks)} != ts_shape {cur_ts_len}"
        )

        pos = 0
        for block_id, bs in enumerate(sample_blocks):
            end = min(pos + bs, cur_ts_len)
            ts_block_ids[ts_cum + pos : ts_cum + end] = block_id
            pos = end

        ts_sample_ids[ts_cum : ts_cum + cur_ts_len] = b_idx
        ts_cum += cur_ts_len

    # 2. Text side: sample_id only
    text_sample_ids = torch.zeros(L_text, dtype=torch.long, device=device)
    text_cum = 0
    for b_idx in range(B):
        cur_text_len = int(text_shape[b_idx].item())
        if cur_text_len == 0:
            continue
        text_sample_ids[text_cum : text_cum + cur_text_len] = b_idx
        text_cum += cur_text_len

    # 3. Global physical layout: [ts ; text]
    global_sample_id = torch.cat([ts_sample_ids, text_sample_ids])  # (L_total,)

    # 4. Sample isolation
    same_sample = (
        global_sample_id.unsqueeze(1) == global_sample_id.unsqueeze(0)
    )  # (L_total, L_total)
    allowed = same_sample.clone()

    # 5. Block-causal on TS->TS region
    ts_causal = (
        ts_block_ids.unsqueeze(1) >= ts_block_ids.unsqueeze(0)
    )  # (L_ts, L_ts)
    allowed[:L_ts, :L_ts] = allowed[:L_ts, :L_ts] & ts_causal

    # 6. Block Text→TS: text rows (Q) cannot attend to TS columns (K)
    allowed[L_ts:, :L_ts] = False

    # TS→Text and Text→Text remain as same_sample allows (within same sample)

    # 7. Build additive mask
    mask = torch.full(
        (L_total, L_total), torch.finfo(dtype).min, dtype=dtype, device=device,
    )
    mask[allowed] = 0.0

    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, L_total, L_total)
