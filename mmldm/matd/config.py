"""MATD default configuration.

Centralises every hyper-parameter for the MATD (Multimodal Adaptive Temporal
Diffusion) framework into a single flat dictionary.  Individual parameter
groups are documented below; each group controls a distinct stage or component
of the pipeline.

Usage::

    from mmldm.matd.config import get_matd_config

    # Defaults
    cfg = get_matd_config()

    # Override specific keys
    cfg = get_matd_config({"dit_depth": 12, "lr": 5e-5})
"""

from __future__ import annotations

from typing import Any, Optional


MATD_DEFAULT_CONFIG: dict[str, Any] = {
    # -----------------------------------------------------------------------
    # Tokenizer (DA-ATP -- Density-Adaptive Adaptive Tokenisation for Patches)
    #
    # Controls the time-series-to-patch-token pipeline.  embed_dim is the
    # dimensionality of each patch latent.  target_tokens is the default
    # number of patches K that the planner produces.  ref_len is the
    # reference sequence length used when computing adaptive patch boundaries.
    # min_len / max_len bound the valid per-patch extent.
    # rfft_win sets the RFFT window size for spectral features inside the
    # tokenizer.  tau is the Gumbel-Softmax temperature for discrete
    # patch-boundary sampling.  base_score is the minimum density score
    # assigned to any time-step to prevent zero-weight patches.
    # -----------------------------------------------------------------------
    "embed_dim": 256,
    "target_tokens": None,
    "ref_len": 16,
    "min_len": 4,
    "max_len": 64,
    "rfft_win": 32,
    "tau": 0.5,
    "base_score": 0.05,
    "target_seg_len": 8,
    "min_tokens": 8,
    "max_tokens": 128,

    # -----------------------------------------------------------------------
    # Text encoder
    #
    # text_model is the HuggingFace identifier for the pretrained
    # sentence-transformer used to encode text captions.  text_dim is its
    # output embedding dimension.  text_frozen freezes the encoder weights
    # so only the optional projection layer is trained.  model_dim is the
    # MATD internal working dimension (matches embed_dim by default) that
    # the text projection targets.
    # -----------------------------------------------------------------------
    "text_model": "sentence-transformers/all-MiniLM-L6-v2",
    "text_dim": 384,
    "text_frozen": True,
    "model_dim": 256,

    # -----------------------------------------------------------------------
    # Semantic slots
    #
    # n_slots is the number of learnable semantic slot vectors used by the
    # SCCI (Semantic Cross-Condition Injection) module to compress the text
    # conditioning into a fixed-size set of concept embeddings.
    # -----------------------------------------------------------------------
    "n_slots": 6,

    # -----------------------------------------------------------------------
    # Planner (Text-to-Patch)
    #
    # planner_heads sets the number of attention heads in the planner
    # cross-attention layer that maps text tokens to patch metadata.
    # -----------------------------------------------------------------------
    "planner_heads": 8,

    # -----------------------------------------------------------------------
    # Mixture-of-Experts (MoE)
    #
    # n_experts is the total number of expert feed-forward networks in
    # the MoE layer.  top_k is the number of experts activated per token
    # during sparse routing (top-k gating).
    # -----------------------------------------------------------------------
    "n_experts": 6,
    "top_k": 2,

    # -----------------------------------------------------------------------
    # Causal discovery (C-SCMON)
    #
    # n_mech is the number of causal mechanism variables modelled by the
    # structural causal module.  n_segments partitions the time axis into
    # this many segments for segment-level causal analysis.  max_lag sets
    # the maximum temporal lag considered when building the causal adjacency
    # matrix.
    # -----------------------------------------------------------------------
    "n_mech": 6,
    "n_segments": 8,
    "max_lag": 2,
    "causal_predict_weight": 1.0,
    "causal_dag_weight": 0.1,
    "causal_sparsity_weight": 0.01,
    "causal_smooth_weight": 0.01,
    "causal_disentangle_weight": 0.01,
    "causal_return_segment_lag_graph": False,

    # -----------------------------------------------------------------------
    # DiT (Diffusion Transformer) denoiser
    #
    # dit_depth is the number of TextTemporalDiTBlock layers stacked in
    # the denoiser.  dit_heads is the number of self- and cross-attention
    # heads per block.  dit_dim is the hidden dimension inside each block.
    # mlp_ratio is the expansion factor for the feed-forward sub-layer
    # (hidden_size = dit_dim * mlp_ratio).  pred_mode selects the
    # diffusion prediction parameterisation: "eps" for epsilon-prediction
    # or "v" for v-prediction.
    # -----------------------------------------------------------------------
    "dit_depth": 8,
    "dit_heads": 8,
    "dit_dim": 256,
    "mlp_ratio": 4.0,
    "pred_mode": "eps",

    # -----------------------------------------------------------------------
    # Decoder (patch latent -> raw time series)
    #
    # decoder_hidden is the hidden-layer width of the MLP decoder that
    # maps patch latents back to the original time-series resolution.
    # -----------------------------------------------------------------------
    "decoder_hidden": 256,

    # -----------------------------------------------------------------------
    # Diffusion schedule
    #
    # timesteps is the total number of diffusion steps used during
    # training.  beta_schedule selects the noise schedule type
    # ("cosine" or "linear").  ddim_steps is the number of steps
    # used by the DDIM fast sampler at inference time.
    # -----------------------------------------------------------------------
    "timesteps": 1000,
    "beta_schedule": "cosine",
    "ddim_steps": 50,

    # -----------------------------------------------------------------------
    # Loss weights
    #
    # Scalar coefficients that balance the multi-objective training loss:
    #   lambda_x0      -- reconstruction (x0 prediction) loss
    #   lambda_delta    -- first-order temporal derivative loss
    #   lambda_fft      -- spectral (RFFT magnitude) loss
    #   lambda_plan     -- planner layout prediction loss
    #   lambda_align    -- text-TS contrastive alignment loss
    #   lambda_moe      -- MoE load-balancing / routing regulariser
    #   lambda_causal   -- causal mechanism / DAG / sparsity loss
    # -----------------------------------------------------------------------
    "lambda_x0": 0.2,
    "lambda_delta": 0.1,
    "lambda_fft": 0.05,
    "lambda_plan": 0.5,
    "lambda_align": 0.05,
    "lambda_moe": 0.01,
    "lambda_causal": 0.01,

    # -----------------------------------------------------------------------
    # Training
    #
    # lr is the peak learning rate (AdamW).  weight_decay is the L2
    # regularisation coefficient.  warmup_steps is the number of linear
    # warm-up steps before cosine decay begins.  total_steps is the total
    # optimiser step budget.  batch_size is the per-GPU mini-batch size.
    # grad_clip clips the global gradient norm to this value.
    # ema_decay is the exponential moving average decay rate for model
    # EMA weights.
    # -----------------------------------------------------------------------
    "lr": 1e-4,
    "weight_decay": 0.01,
    "warmup_steps": 1000,
    "total_steps": 100000,
    "batch_size": 32,
    "grad_clip": 1.0,
    "ema_decay": 0.9999,

    # -----------------------------------------------------------------------
    # Classifier-Free Guidance (CFG)
    #
    # p_drop_text is the probability of replacing the text conditioning
    # with null embeddings during training (enables CFG at inference).
    # cfg_scale is the guidance scale multiplier applied to the
    # conditional prediction at inference time:
    #     eps_guided = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
    # -----------------------------------------------------------------------
    "p_drop_text": 0.1,
    "cfg_scale": 5.0,

    # -----------------------------------------------------------------------
    # Staged meta training
    #
    # use_oracle_meta_prob is the probability of using oracle tokenizer
    # metadata instead of planner predictions during training.  Start at
    # 1.0 (all oracle) and anneal to 0.0 over training for stable
    # planner learning before end-to-end fine-tuning.
    # -----------------------------------------------------------------------
    "use_oracle_meta_prob": 1.0,
}


def get_matd_config(overrides: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Return the MATD configuration with optional per-key overrides.

    Args:
        overrides: Dictionary of keys to override in the default config.
            Only keys present in this dict are replaced; all others keep
            their default values.

    Returns:
        A new dictionary containing the merged configuration.

    Example::

        cfg = get_matd_config({"dit_depth": 12, "lr": 5e-5})
        assert cfg["dit_depth"] == 12
        assert cfg["lr"] == 5e-5
        assert cfg["batch_size"] == 32  # unchanged default
    """
    config = MATD_DEFAULT_CONFIG.copy()
    if overrides is not None:
        config.update(overrides)
    return config
