"""Image Encoder for TIGER — encode 3-channel images (GAF+STFT+RP) into embeddings.

Provides two encoder variants:
  - ImageEncoder: CLIP ViT (frozen) + learned projection MLP (recommended)
  - CNNEncoder:    Lightweight 4-layer CNN (fallback when CLIP is unavailable)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List


class ImageEncoder(nn.Module):
    """Encode 3-channel images using CLIP ViT + learned projection MLP.

    Architecture mirrors VerbalTS CLIPTextEncoder but swaps the text encoder
    for the vision encoder: frozen CLIPVisionModel produces patch-level hidden
    states, then a learned MLP projects them to the target embedding dimension.

    Parameters
    ----------
    configs : dict
        Required keys:
            pretrain_model_path : str
                HuggingFace model id or local path for CLIP ViT
                (e.g. "openai/clip-vit-base-patch32").
            pretrain_model_dim : int
                Hidden dimension of the CLIP vision model
                (e.g. 768 for ViT-B/32, 1024 for ViT-L/14).
            imageemb_hidden_dim : int
                Hidden dimension of the projection MLP.
            image_emb : int
                Output embedding dimension.
            device : str or torch.device
                Device to place the frozen model on.
    """

    def __init__(self, configs: dict):
        super().__init__()
        self.configs = configs
        self.device = configs["device"]
        self.emb_dim = configs["image_emb"]

        # ------------------------------------------------------------------
        # Frozen CLIP vision backbone
        # ------------------------------------------------------------------
        from transformers import CLIPVisionModel

        self.model = CLIPVisionModel.from_pretrained(configs["pretrain_model_path"])
        for param in self.model.parameters():
            param.requires_grad = False

        self.hidden_dim = self.model.config.hidden_size  # sanity-check alias

        # ------------------------------------------------------------------
        # Learned projection MLP  (mirrors VerbalTS text_enc)
        # ------------------------------------------------------------------
        self.image_enc = nn.Sequential(
            nn.Linear(configs["pretrain_model_dim"], configs["imageemb_hidden_dim"]),
            nn.LayerNorm(configs["imageemb_hidden_dim"]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(configs["imageemb_hidden_dim"], configs["image_emb"]),
        )

    def forward(
        self,
        images: Union[torch.Tensor, List],
    ) -> torch.Tensor:
        """Encode images into token-level embeddings.

        Parameters
        ----------
        images : torch.Tensor or list
            Either a float tensor of shape ``(B, 3, H, W)`` already on the
            correct device, or a list of PIL images / numpy arrays that will
            be processed by the CLIP image processor.

        Returns
        -------
        torch.Tensor
            Shape ``(B, N, image_emb)`` where ``N = num_patches + 1``
            (all patch tokens plus the CLS token).
        """
        if isinstance(images, torch.Tensor):
            # Expect (B, 3, H, W) float tensor
            outputs = self.model(pixel_values=images.to(self.device))
        else:
            # List of PIL images — apply the CLIP image processor
            from transformers import CLIPProcessor

            processor = CLIPProcessor.from_pretrained(
                self.configs["pretrain_model_path"]
            )
            inputs = processor(images=images, return_tensors="pt")
            outputs = self.model(**{k: v.to(self.device) for k, v in inputs.items()})

        # Use ALL hidden states (CLS + all patch tokens)
        # last_hidden_state: (B, N, hidden_dim)  where N = num_patches + 1
        hidden_states = outputs.last_hidden_state  # (B, N, pretrain_model_dim)

        # Project every token independently
        image_emb = self.image_enc(hidden_states)  # (B, N, image_emb)

        return image_emb


class CNNEncoder(nn.Module):
    """Lightweight CNN encoder for 3-channel images.

    A simple 4-layer convolutional network that can serve as a drop-in
    replacement for :class:`ImageEncoder` when CLIP is unavailable.
    Produces a spatial grid of embeddings rather than a variable-length
    token sequence, making it fully deterministic in output length.

    Parameters
    ----------
    configs : dict
        Required keys:
            image_emb : int
                Output embedding dimension per spatial location.
    """

    def __init__(self, configs: dict):
        super().__init__()
        out_dim = configs["image_emb"]

        self.features = nn.Sequential(
            # Block 1: 3 -> 16
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            # Block 2: 16 -> 32
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # Block 3: 32 -> 64
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Block 4: 64 -> 128
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # Pool to a fixed spatial size and project to output dim
        self.pool = nn.AdaptiveAvgPool2d((4, 4))  # always 4x4 = 16 tokens
        self.proj = nn.Linear(128, out_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images into a fixed-length embedding sequence.

        Parameters
        ----------
        images : torch.Tensor
            Float tensor of shape ``(B, 3, H, W)``.

        Returns
        -------
        torch.Tensor
            Shape ``(B, 16, image_emb)`` — 16 spatial tokens.
        """
        feat = self.features(images)          # (B, 128, H', W')
        feat = self.pool(feat)                # (B, 128, 4, 4)
        feat = feat.flatten(2).transpose(1, 2)  # (B, 16, 128)
        out = self.proj(feat)                 # (B, 16, image_emb)
        return out
