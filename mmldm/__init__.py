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

"""MMLDM: Multimodal Latent Diffusion Model for Time Series Generation.

This package implements the MMLDM framework, combining:
- Cola-DLM's block-causal continuous latent diffusion
- MMDiT's JointAttention multimodal fusion
- DCD (Dual-Condition Denoising) for mixed-sample training
- Adaptive Semantic Patching for text-guided dynamic block allocation
"""

from .attention_utils import create_dit_readonly_text_mask, create_multimodal_joint_mask
from .configuration_mmldm import MMLDMDiTConfig, MMLDMVAEConfig

__all__ = [
    "MMLDMVAEConfig",
    "MMLDMDiTConfig",
    "create_dit_readonly_text_mask",
    "create_multimodal_joint_mask",
]

# Lazy imports for heavy modules
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "MMLDMVAEModel": (".modeling_mmldm_vae", "MMLDMVAEModel"),
    "MMLDMDiTModel": (".modeling_mmldm_dit", "MMLDMDiTModel"),
    "SemanticRouter": (".semantic_router", "SemanticRouter"),
    "generate_timeseries": (".inference", "generate_timeseries"),
    "evaluate_single": (".evaluation", "evaluate_single"),
    "evaluate_multi": (".evaluation", "evaluate_multi"),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        import importlib

        module = importlib.import_module(module_path, __name__)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(__all__) + list(_LAZY_IMPORTS.keys())
