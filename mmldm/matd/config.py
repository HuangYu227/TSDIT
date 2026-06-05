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
    "encoder_context_depth": 2,
    "encoder_num_heads": 8,
    "encoder_spectral_bins": 8,
    "encoder_meta_fourier_bands": 4,

    # -----------------------------------------------------------------------
    # Text encoder
    # -----------------------------------------------------------------------
    "text_model": "sentence-transformers/all-MiniLM-L6-v2",
    "text_dim": 384,
    "text_frozen": True,
    "model_dim": 256,
    "text_max_length": 128,

    # -----------------------------------------------------------------------
    # Planner / SCCI
    # -----------------------------------------------------------------------
    "n_slots": 6,
    "slot_iters": 2,
    "planner_heads": 8,
    "scci_heads": 4,
    "scci_dropout": 0.0,

    # -----------------------------------------------------------------------
    # Mixture-of-Experts (MoE)
    # -----------------------------------------------------------------------
    "n_experts": 6,
    "top_k": 2,
    "moe_hidden_mult": 4,
    "moe_router_hidden_mult": 2,
    "moe_dropout": 0.0,
    "moe_noisy_gating": True,
    "moe_capacity_factor_train": 1.25,
    "moe_capacity_factor_eval": 2.0,
    "moe_prior_scale": 1.0,
    "moe_balance_weight": 0.01,
    "moe_z_loss_weight": 0.001,
    "moe_prior_kl_weight": 0.05,
    "moe_router_smooth_weight": 0.01,
    "moe_capacity_weight": 0.1,
    "moe_residual_scale": 0.1,

    # -----------------------------------------------------------------------
    # Causal discovery (C-SCMON)
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
    # -----------------------------------------------------------------------
    "dit_depth": 8,
    "dit_heads": 8,
    "dit_dim": 256,
    "mlp_ratio": 4.0,
    "pred_mode": "eps",
    "dit_dropout": 0.0,
    "dit_qk_norm": False,
    "min_snr_gamma": 5.0,
    "eval_interval": 10,
    "log_interval": 50,

    # -----------------------------------------------------------------------
    # Decoder (patch latent -> raw time series)
    # -----------------------------------------------------------------------
    "decoder_hidden": 256,
    "decoder_context_depth": 1,
    "decoder_context_heads": 8,
    "decoder_local_bands": 8,
    "decoder_global_bands": 6,
    "decoder_chunk_size": 16,
    "decoder_field_blocks": 3,
    "decoder_siren_omega": 18.0,
    "decoder_siren_scale": 0.1,
    "decoder_output_activation": "none",

    # -----------------------------------------------------------------------
    # Diffusion schedule
    # -----------------------------------------------------------------------
    "timesteps": 1000,
    "beta_schedule": "cosine",
    "ddim_steps": 50,

    # -----------------------------------------------------------------------
    # Loss weights
    # -----------------------------------------------------------------------
    "lambda_diffusion": 1.0,
    "lambda_x0": 0.2,
    "lambda_delta": 0.1,
    "lambda_fft": 0.05,
    "lambda_plan": 0.5,
    "lambda_align": 0.05,
    "lambda_moe": 0.01,
    "lambda_causal": 0.01,
    "lambda_scci": 0.0,
    "lambda_latent_anchor": 0.01,

    # -----------------------------------------------------------------------
    # Training
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
    # -----------------------------------------------------------------------
    "p_drop_text": 0.1,
    "cfg_scale": 5.0,

    # -----------------------------------------------------------------------
    # Staged meta training
    # -----------------------------------------------------------------------
    "use_oracle_meta_prob": 1.0,

    # -----------------------------------------------------------------------
    # Sampling
    # -----------------------------------------------------------------------
    "use_causal_guidance_in_sampling": True,
    "planner_beta": 1.0,
    "align_temperature": 0.07,
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
