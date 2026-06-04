"""MATD -- Multimodal Adaptive Temporal Diffusion framework."""

# Config
from .config import MATD_DEFAULT_CONFIG, get_matd_config

# Data
from .data_adapter import MATDDataset, MATDDataModule, matd_collate_fn

# Tokenizer (DA-ATP)
from .tokenizer import (
    DensityAwareAdaptivePatch,
    TemporalContextBlock,
    AdaptiveTemporalEncoder,
)

# Text encoder
from .text_encoder import MATDTextEncoder, NullTextEncoder

# Planner
from .planner import TextToPatchPlanner, PlannerLoss

# SCCI
from .scci import TextSemanticSlotExtractor, SemanticCausalConditionInjector

# MoE
from .moe import SemanticCausalTemporalMoE

# Causal
from .causal import DynamicCausalMechanismLearner

# DiT
from .dit import T2PDenoiser, TextTemporalDiTBlock, RelativeTemporalSelfAttention

# Decoder
from .decoder import VariablePatchDecoder, LinearPatchDecoder

# Losses
from .losses import (
    AdaptivePatchFieldLoss,
    CausalSemanticRouterLoss,
    LatentDiffusionBridgeLoss,
    TextLayoutLoss,
    LossWeights,
    compute_total_loss,
    weights_from_config,
)

# Full model
from .matd_model import MATDModel, MATDConfig

# Trainer & Generator
from .trainer import MATDTrainer, EMA
from .generator import MATDGenerator

# Evaluator
from .evaluator import MATDEvaluator, calculate_mrr

__all__ = [
    "MATD_DEFAULT_CONFIG", "get_matd_config",
    "MATDDataset", "MATDDataModule", "matd_collate_fn",
    "DensityAwareAdaptivePatch", "TemporalContextBlock", "AdaptiveTemporalEncoder",
    "MATDTextEncoder", "NullTextEncoder",
    "TextToPatchPlanner", "PlannerLoss",
    "TextSemanticSlotExtractor", "SemanticCausalConditionInjector",
    "SemanticCausalTemporalMoE",
    "DynamicCausalMechanismLearner",
    "T2PDenoiser", "TextTemporalDiTBlock", "RelativeTemporalSelfAttention",
    "VariablePatchDecoder", "LinearPatchDecoder",
    "LatentDiffusionBridgeLoss", "AdaptivePatchFieldLoss", "TextLayoutLoss",
    "CausalSemanticRouterLoss", "LossWeights", "compute_total_loss", "weights_from_config",
    "MATDModel", "MATDConfig",
    "MATDTrainer", "EMA", "MATDGenerator",
    "MATDEvaluator", "calculate_mrr",
]
