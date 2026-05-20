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
    * ``text_dim``: dimension of the text embedding input (128 for TSFragment-600K).
    * ``dim``: hidden dimension of the transformer trunk.
    * ``latent_dim``: dimension ``d`` of the continuous latent ``z_0``.
    * ``patch_size``: 1-D patchification factor (1 for short time series).
    * ``encoder_num_blocks`` / ``decoder_num_blocks`` / ``joint_num_blocks``:
      depth of TS encoder, decoder, and joint encoder respectively.
    * ``block_size``: default block size for block-causal attention.
    * ``use_variation``: whether to use Gaussian posterior (mean + logvar).

    Defaults: d=128, latent_dim=16, 4 encoder + 4 decoder + 2 joint blocks.
    """

    model_type = "mmldm_vae"

    def __init__(
        self,
        ts_channels: int = 1,
        text_dim: int = 128,
        dim: int = 128,
        ffn_dim: int = 512,
        latent_dim: int = 16,
        patch_size: int = 1,
        num_heads: int = 4,
        head_dim: int = 32,
        shared_heads_kv: int = 1,
        encoder_num_blocks: int = 4,
        decoder_num_blocks: int = 4,
        joint_num_blocks: int = 2,
        layer_norm_eps: float = 1e-6,
        post_norm: bool = True,
        qk_bias: bool = False,
        qk_norm: bool = True,
        rope_theta: int = 10000,
        bias: bool = True,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        act: str = "swiglu",
        block_causal: bool = True,
        block_size: int = 4,
        init_fn: str = "normal",
        init_std: float = 0.02,
        init_cutoff_factor: float = 3,
        use_variation: bool = True,
        scaling_factor: float = 1.0,
        shifting_factor: float = 0.0,
        **kwargs,
    ):
        self.ts_channels = ts_channels
        self.text_dim = text_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.latent_dim = latent_dim
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.shared_heads_kv = shared_heads_kv
        self.encoder_num_blocks = encoder_num_blocks
        self.decoder_num_blocks = decoder_num_blocks
        self.joint_num_blocks = joint_num_blocks
        self.layer_norm_eps = layer_norm_eps
        self.post_norm = post_norm
        self.qk_bias = qk_bias
        self.qk_norm = qk_norm
        self.rope_theta = rope_theta
        self.bias = bias
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.act = act
        self.block_causal = block_causal
        self.block_size = block_size
        self.init_fn = init_fn
        self.init_std = init_std
        self.init_cutoff_factor = init_cutoff_factor
        self.use_variation = use_variation
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
        ts_in_channels: int = 16,
        ts_out_channels: int = 16,
        text_in_channels: int = 16,
        text_out_channels: int = 16,
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
        block_size: int = 4,
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
