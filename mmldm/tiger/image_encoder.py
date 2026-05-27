"""Image Encoder for TIGER — encode 3-channel images (GAF+STFT+RP) into embeddings.

Provides three encoder variants:
  - ImageEncoder:  CLIP ViT (frozen) + learned projection MLP
  - ViTEncoder:    Lightweight Vision Transformer (trainable, recommended)
  - CNNEncoder:    Lightweight 4-layer CNN (fallback)
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
            # Expect (B, 3, H, W) float tensor in [0, 1].
            # CLIP expects 224x224 with specific normalization.
            x = images.to(self.device)
            if x.shape[-1] != 224 or x.shape[-2] != 224:
                x = F.interpolate(x, size=(224, 224), mode="bicubic", align_corners=False)
            # CLIP image normalization (ImageNet stats)
            mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=x.device).view(1, 3, 1, 1)
            std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=x.device).view(1, 3, 1, 1)
            x = (x - mean) / std
            outputs = self.model(pixel_values=x)
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


class PatchEmbed(nn.Module):
    """Split image into patches and project to embedding dimension."""

    def __init__(self, img_size=64, patch_size=8, in_chans=3, embed_dim=192):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=patch_size)

    def forward(self, x):
        # (B, 3, H, W) -> (B, embed_dim, H/P, W/P) -> (B, num_patches, embed_dim)
        return self.proj(x).flatten(2).transpose(1, 2)


class TransformerBlock(nn.Module):
    """Standard pre-norm Transformer block with GELU activation."""

    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=drop,
                                           batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(drop),
        )

    def forward(self, x):
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class ViTEncoder(nn.Module):
    """Lightweight Vision Transformer for 3-channel images.

    Designed for 64×64 spectrogram images. Splits into 8×8 patches
    (64 tokens), applies Transformer self-attention, and outputs a
    token sequence compatible with ``_AnchorProjector``.

    Parameters
    ----------
    configs : dict
        Required keys:
            image_emb : int
                Output embedding dimension per token.
        Optional keys:
            img_size  : int, default 64
            patch_size: int, default 8
            embed_dim : int, default 192
            depth     : int, default 4  (number of Transformer layers)
            num_heads : int, default 6
            mlp_ratio : float, default 4.0
            drop      : float, default 0.1
    """

    def __init__(self, configs: dict):
        super().__init__()
        img_size = configs.get("img_size", 64)
        patch_size = configs.get("patch_size", 8)
        embed_dim = configs.get("embed_dim", 192)
        depth = configs.get("depth", 4)
        num_heads = configs.get("num_heads", 6)
        mlp_ratio = configs.get("mlp_ratio", 4.0)
        drop = configs.get("drop", 0.1)
        out_dim = configs["image_emb"]

        self.patch_embed = PatchEmbed(img_size, patch_size, 3, embed_dim)
        num_patches = self.patch_embed.num_patches

        # CLS token + positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + num_patches, embed_dim))
        self.pos_drop = nn.Dropout(drop)

        # Transformer blocks
        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, mlp_ratio, drop)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Projection to output dim
        self.proj = nn.Linear(embed_dim, out_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images into a token embedding sequence.

        Parameters
        ----------
        images : torch.Tensor
            Float tensor of shape ``(B, 3, H, W)``.

        Returns
        -------
        torch.Tensor
            Shape ``(B, 1 + num_patches, image_emb)`` — CLS + patch tokens.
        """
        B = images.shape[0]
        x = self.patch_embed(images)  # (B, num_patches, embed_dim)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, 1+num_patches, embed_dim)

        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)

        return self.proj(x)  # (B, 1+num_patches, image_emb)


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
