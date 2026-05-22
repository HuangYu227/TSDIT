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

"""Configuration classes for MMLDM VAE and DiT models."""

from transformers import PretrainedConfig


class MMLDMVAEConfig(PretrainedConfig):
    """Configuration for :class:`MMLDMVAEModel`.

    Parameterizes the **Multimodal VAE** of MMLDM, which provides the
    shared latent space for time series and text, the inference encoder
    ``q_phi(z_0 | x, c)``, and the conditional decoder
    ``p_theta(x | z_0)``.

    Key knobs:
    * ``ts_channels``: number of channels in the input time series (1 for univariate).
    * ``dim``: hidden dimension of the Conv1d encoder/decoder trunk.
    * ``latent_dim``: dimension ``d`` of the continuous latent ``z_0``.
    * ``num_conv_layers``: number of Conv1d + Residual blocks in encoder/decoder.
    * ``encoder_num_blocks`` / ``decoder_num_blocks``: depth of encoder/decoder.
    * ``block_size``: default block size for block-causal attention.
    * ``use_variation``: whether to use Gaussian posterior (mean + logvar).
    * ``kl_anneal_start`` / ``kl_anneal_end`` / ``kl_anneal_epochs``: KL weight schedule.

    Defaults: d=128, latent_dim=64, 4 encoder + 4 decoder blocks.
    """

    model_type = "mmldm_vae"

    def __init__(
        self,
        ts_channels: int = 1,
        dim: int = 128,
        ffn_dim: int = 512,
        latent_dim: int = 64,
        text_dim: int = 128,
        patch_size: int = 1,
        num_conv_layers: int = 3,
        num_heads: int = 4,
        head_dim: int = 32,
        encoder_num_blocks: int = 4,
        decoder_num_blocks: int = 4,
        layer_norm_eps: float = 1e-6,
        bias: bool = True,
        dropout: float = 0.0,
        block_causal: bool = True,
        block_size: int = 8,
        use_variation: bool = True,
        kl_anneal_start: float = 0.0,
        kl_anneal_end: float = 1e-6,
        kl_anneal_epochs: int = 5,
        fft_cutoff_ratio: float = 0.3,
        scaling_factor: float = 1.0,
        shifting_factor: float = 0.0,
        **kwargs,
    ):
        self.ts_channels = ts_channels
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.latent_dim = latent_dim
        self.text_dim = text_dim
        self.patch_size = patch_size
        self.num_conv_layers = num_conv_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.encoder_num_blocks = encoder_num_blocks
        self.decoder_num_blocks = decoder_num_blocks
        self.layer_norm_eps = layer_norm_eps
        self.bias = bias
        self.dropout = dropout
        self.block_causal = block_causal
        self.block_size = block_size
        self.use_variation = use_variation
        self.kl_anneal_start = kl_anneal_start
        self.kl_anneal_end = kl_anneal_end
        self.kl_anneal_epochs = kl_anneal_epochs
        self.fft_cutoff_ratio = fft_cutoff_ratio
        self.scaling_factor = scaling_factor
        self.shifting_factor = shifting_factor
        super().__init__(**kwargs)


class MMLDMDiTConfig(PretrainedConfig):
    """Configuration for :class:`MMLDMDiTModel`.

    Parameterizes the **Multimodal DiT prior** ``p_psi(z_0 | c)`` of MMLDM.
    The DiT learns the vector field ``v_psi(z_t, t; z_0^{(<b)}, c)``
    under the visible set ``V_b = {sg(z_0^{(<b)}), z_t^(b), c}``.

    Key knobs:
    * ``ts_in_channels`` / ``ts_out_channels``: latent dim for time series modality.
    * ``text_in_channels`` / ``text_out_channels``: latent dim for text modality.
    * ``txt_dim``: hidden width of the DiT trunk.
    * ``emb_dim``: AdaLN timestep embedding dimension.
    * ``heads`` / ``head_dim``: attention shape.
    * ``num_layers``: depth of the DiT trunk.
    * ``block_size``: default block size for block-causal attention.
    * ``rope_dim``: number of channels per head that receive RoPE.

    Defaults: 12 layers, 4 heads, head_dim=32, txt_dim=256.
    """

    model_type = "mmldm_dit"

    def __init__(
        self,
        ts_in_channels: int = 64,
        ts_out_channels: int = 64,
        text_in_channels: int = 64,
        text_out_channels: int = 64,
        txt_dim: int = 256,
        emb_dim: int = 256,
        heads: int = 4,
        head_dim: int = 32,
        expand_ratio: int = 4,
        num_layers: int = 12,
        norm_eps: float = 1e-5,
        qk_bias: bool = False,
        patch_size: int = 1,
        rope_dim: int = 32,
        block_size: int = 8,
        **kwargs,
    ):
        self.ts_in_channels = ts_in_channels
        self.ts_out_channels = ts_out_channels
        self.text_in_channels = text_in_channels
        self.text_out_channels = text_out_channels
        self.txt_dim = txt_dim
        self.emb_dim = emb_dim
        self.heads = heads
        self.head_dim = head_dim
        self.expand_ratio = expand_ratio
        self.num_layers = num_layers
        self.norm_eps = norm_eps
        self.qk_bias = qk_bias
        self.patch_size = patch_size
        self.rope_dim = rope_dim
        self.block_size = block_size
        super().__init__(**kwargs)
