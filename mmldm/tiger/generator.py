"""TIGER generator.

Paper-ready changes:
  1. True text+image conditioning through ``MultiModalConditioner``.
  2. Self-supervised masked reference-image construction when datasets do not
     provide explicit ``ref_image`` / ``cond_image`` fields.
  3. Multimodal classifier-free guidance with text, image, and interaction
     guidance terms.
  4. CTICD receives the clean target image during training, so causal graphs are
     not learned only from noisy diffusion states.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .dit_model import TIGERDiT
from .cond_projector import TextOnlyProjector, MultiModalConditioner
from .samplers import DDPMSampler, DDIMSampler


class TIGERGenerator(nn.Module):
    """Diffusion generator for TS-images conditioned on text and optional images."""

    def __init__(self, config):
        super().__init__()
        self.device = config["device"]
        self.config = config

        diff_config = config["diffusion"]
        cond_config = config["condition"]
        self.cond_mode = str(cond_config.get("cond_mode", "text_image")).lower()

        self._init_text_encoder(cond_config)
        self._init_cond_projector(diff_config, cond_config)
        self._init_dit(diff_config)

        self.num_steps = diff_config["num_steps"]
        self.ddpm = DDPMSampler(
            self.num_steps,
            diff_config["beta_start"],
            diff_config["beta_end"],
            diff_config.get("schedule", "quad"),
            self.device,
        )
        self.ddim = DDIMSampler(
            self.num_steps,
            diff_config["beta_start"],
            diff_config["beta_end"],
            diff_config.get("schedule", "quad"),
            self.device,
        )

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_text_encoder(self, cond_config):
        from transformers import AutoTokenizer, CLIPTextConfig, CLIPTextModelWithProjection

        model_path = cond_config["text"].get("pretrain_model_path", "openai/clip-vit-base-patch32")
        self.text_tokenizer = AutoTokenizer.from_pretrained(model_path)
        if "Long" in model_path or "longclip" in model_path.lower():
            clip_config = CLIPTextConfig.from_pretrained(model_path)
            clip_config.max_position_embeddings = 248
            self.text_model = CLIPTextModelWithProjection.from_pretrained(model_path, config=clip_config).to(self.device)
        else:
            self.text_model = CLIPTextModelWithProjection.from_pretrained(model_path).to(self.device)
        for param in self.text_model.parameters():
            param.requires_grad = False
        self.text_proj = nn.Sequential(
            nn.Linear(cond_config["text"]["pretrain_model_dim"], cond_config["text"]["textemb_hidden_dim"]),
            nn.LayerNorm(cond_config["text"]["textemb_hidden_dim"]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(cond_config["text"]["textemb_hidden_dim"], cond_config["text"]["text_emb"]),
        ).to(self.device)

    def _init_cond_projector(self, diff_config, cond_config):
        n_var = diff_config.get("n_var", 16)
        n_scale = diff_config.get("multipatch_num", 1)
        n_steps = diff_config["num_steps"]
        n_stages = cond_config.get("num_stages", 4)

        if self.cond_mode in {"text_image", "multimodal", "image_text"}:
            image_cfg = dict(cond_config.get("image", {}))
            image_cfg.setdefault("device", self.device)
            image_cfg.setdefault("img_size", diff_config.get("image_size", self.config.get("image_size", 64)))
            image_cfg.setdefault("image_emb", cond_config.get("joint_emb", diff_config["channels"]))
            self.cond_projector = MultiModalConditioner(
                n_var=n_var,
                n_scale=n_scale,
                n_steps=n_steps,
                n_stages=n_stages,
                dim_in_text=cond_config["text"]["text_emb"],
                dim_out=diff_config["channels"],
                dim_joint=cond_config.get("joint_emb", diff_config["channels"]),
                image_cfg=image_cfg,
                n_heads=cond_config.get("fusion_heads", 8),
                n_layers=cond_config.get("fusion_layers", 2),
                dim_ff=cond_config.get("fusion_ff", max(256, 4 * diff_config["channels"])),
            ).to(self.device)
        elif self.cond_mode in {"text_only", "text"}:
            self.cond_projector = TextOnlyProjector(
                n_var=n_var,
                n_scale=n_scale,
                n_steps=n_steps,
                n_stages=n_stages,
                dim_in_text=cond_config["text"]["text_emb"],
                dim_out=diff_config["channels"],
            ).to(self.device)
        else:
            raise ValueError(f"Unsupported condition mode: {self.cond_mode}")

    def _init_dit(self, diff_config):
        self.dit = TIGERDiT(diff_config).to(self.device)

    # ------------------------------------------------------------------
    # Conditioning utilities
    # ------------------------------------------------------------------

    @property
    def is_multimodal(self) -> bool:
        return self.cond_mode in {"text_image", "multimodal", "image_text"}

    def encode_text(self, texts):
        """Encode text descriptions to token-level embeddings."""
        if isinstance(texts, str):
            texts = [texts]
        max_len = self.text_model.config.max_position_embeddings
        inputs = self.text_tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        text_hidden = self.text_model(**inputs).last_hidden_state
        return self.text_proj(text_hidden)

    def build_reference_image(self, images: torch.Tensor, is_train: bool = True) -> torch.Tensor:
        """Create a degraded reference image if the dataset lacks one.

        The default ``masked_self`` mode turns existing unconditional text-to-TS
        datasets into text+image editing/inpainting datasets without leaking the
        full target image.  It masks contiguous time-axis bands and optionally
        adds small noise.  Real paired editing datasets can override this by
        putting ``ref_image``, ``cond_image``, or ``partial_image`` in the batch.
        """
        cfg = self.config.get("condition", {}).get("reference", {})
        mode = str(cfg.get("mode", "masked_self")).lower()
        if mode in {"none", "null"}:
            return None
        if mode == "identity":
            return images.detach()
        if mode != "masked_self":
            raise ValueError(f"Unknown reference mode: {mode}")

        ref = images.detach().clone()
        B, _C, H, W = ref.shape
        mask_ratio = float(cfg.get("mask_ratio", 0.5))
        mask_ratio = min(max(mask_ratio, 0.0), 0.95)
        noise_std = float(cfg.get("noise_std", 0.02 if is_train else 0.0))
        fill = float(cfg.get("fill", 0.5))
        mask = torch.ones(B, 1, H, W, device=ref.device, dtype=ref.dtype)
        width = max(1, int(round(W * mask_ratio)))
        for b in range(B):
            if is_train:
                start = torch.randint(0, max(1, W - width + 1), (1,), device=ref.device).item()
            else:
                start = max(0, (W - width) // 2)
            end = min(W, start + width)
            mask[b, :, :, start:end] = 0.0
        ref = ref * mask + fill * (1.0 - mask)
        if noise_std > 0:
            ref = ref + noise_std * torch.randn_like(ref)
        return ref.clamp(0.0, 1.0)

    def _batch_reference(self, batch: dict, images: torch.Tensor, is_train: bool) -> Optional[torch.Tensor]:
        for key in ("ref_image", "cond_image", "partial_image", "source_image"):
            if key in batch and batch[key] is not None:
                return batch[key].to(self.device).float()
        if self.is_multimodal:
            return self.build_reference_image(images, is_train=is_train)
        return None

    def _sample_modality_dropout(self, B: int) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.training or not self.is_multimodal:
            return None, None
        cfg = self.config.get("condition", {})
        device = self.device
        p_text = float(cfg.get("drop_text_prob", 0.10))
        p_image = float(cfg.get("drop_image_prob", 0.10))
        p_both = float(cfg.get("drop_both_prob", cfg.get("cfg_dropout", 0.10)))
        drop_text = torch.rand(B, device=device) < p_text
        drop_image = torch.rand(B, device=device) < p_image
        drop_both = torch.rand(B, device=device) < p_both
        drop_text = drop_text | drop_both
        drop_image = drop_image | drop_both
        return drop_text, drop_image

    def compute_condition(
        self,
        texts,
        batch_size: int,
        diffusion_step: torch.Tensor,
        ref_image: Optional[torch.Tensor] = None,
        drop_text_mask: Optional[torch.Tensor] = None,
        drop_image_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.is_multimodal:
            text_emb = None if texts is None else self.encode_text(texts)
            if texts is None and drop_text_mask is None:
                drop_text_mask = torch.ones(batch_size, dtype=torch.bool, device=self.device)
            if ref_image is None and drop_image_mask is None:
                drop_image_mask = torch.ones(batch_size, dtype=torch.bool, device=self.device)
            return self.cond_projector(
                text_emb=text_emb,
                diffusion_step=diffusion_step,
                ref_image=ref_image,
                drop_text_mask=drop_text_mask,
                drop_image_mask=drop_image_mask,
            )

        # Text-only baseline path.
        if texts is not None:
            text_emb = self.encode_text(texts)
        else:
            text_emb = self.encode_text([""] * batch_size)
        return self.cond_projector(text_emb, diffusion_step)

    def compute_condition_from_emb(
        self,
        text_emb: Optional[torch.Tensor],
        diffusion_step: torch.Tensor,
        ref_image: Optional[torch.Tensor] = None,
        drop_text_mask: Optional[torch.Tensor] = None,
        drop_image_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = diffusion_step.shape[0]
        if self.is_multimodal:
            return self.cond_projector(
                text_emb=text_emb,
                diffusion_step=diffusion_step,
                ref_image=ref_image,
                drop_text_mask=drop_text_mask,
                drop_image_mask=drop_image_mask,
            )
        if text_emb is None:
            text_emb = self.encode_text([""] * B)
        return self.cond_projector(text_emb, diffusion_step)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _noise_estimation_loss(self, x, attr_emb, t):
        noise = torch.randn_like(x)
        noisy_x = self.ddpm.forward(x, t, noise)

        pred_noise = self.dit(noisy_x, t, attr_emb, clean_image=x)
        residual = noise - pred_noise
        noise_loss = (residual ** 2).mean()

        loss = noise_loss
        loss_dict = {"noise_loss": noise_loss.detach()}

        if hasattr(self.dit, "_cticd_losses") and self.dit._cticd_losses is not None:
            lambda_cticd = self.config["diffusion"].get("lambda_cticd", 0.1)
            cticd_total = self.dit._cticd_losses["cticd_total"]
            loss = loss + lambda_cticd * cticd_total
            loss_dict["cticd_weighted"] = (lambda_cticd * cticd_total).detach()
            for k, v in self.dit._cticd_losses.items():
                loss_dict[k] = v.detach() if torch.is_tensor(v) else v

        if hasattr(self.dit, "_csa_moe_losses") and self.dit._csa_moe_losses is not None:
            lambda_moe = self.config["diffusion"].get("lambda_moe", 0.1)
            loss = loss + lambda_moe * self.dit._csa_moe_losses
            loss_dict["moe_aux"] = self.dit._csa_moe_losses.detach()

        loss_dict["all"] = loss
        return loss_dict

    def forward(self, batch, is_train=True):
        images = batch["image"].to(self.device).float()
        texts = batch.get("cap", None)
        B = images.shape[0]
        t = torch.randint(0, self.num_steps, [B], device=self.device)

        ref_image = self._batch_reference(batch, images, is_train=is_train)
        drop_text_mask, drop_image_mask = self._sample_modality_dropout(B)

        # Backward-compatible text CFG for text-only baseline.
        if is_train and not self.is_multimodal:
            cfg_dropout = self.config["condition"].get("cfg_dropout", 0.0)
            if cfg_dropout > 0 and texts is not None:
                mask = torch.rand(B, device=self.device) < cfg_dropout
                if mask.any():
                    texts = [txt if not bool(m.item()) else "" for txt, m in zip(texts, mask)]

        attr_emb = self.compute_condition(
            texts,
            B,
            t,
            ref_image=ref_image,
            drop_text_mask=drop_text_mask,
            drop_image_mask=drop_image_mask,
        )
        return self._noise_estimation_loss(images, attr_emb, t)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        image_shape,
        texts,
        n_samples: int = 1,
        sampler: str = "ddim",
        guidance_scale: float = 1.0,
        ref_images: Optional[torch.Tensor] = None,
        image_guidance_scale: Optional[float] = None,
        interaction_guidance_scale: float = 0.0,
        intervention: Optional[dict] = None,
    ):
        """Generate TS-images.

        For multimodal conditioning, ``guidance_scale`` controls text guidance,
        ``image_guidance_scale`` controls image guidance, and
        ``interaction_guidance_scale`` controls the extra text-image interaction
        term.  Setting all three to 1/1/0 gives standard conditional sampling.
        """
        B = len(texts) if texts is not None else (ref_images.shape[0] if ref_images is not None else 1)
        sample_shape = (B, *image_shape)
        samples = []
        ref_images = None if ref_images is None else ref_images.to(self.device).float()

        text_emb = None if texts is None else self.encode_text(texts)
        empty_text_emb = None if self.is_multimodal else self.encode_text([""] * B)
        img_w = guidance_scale if image_guidance_scale is None else image_guidance_scale

        for _ in range(n_samples):
            x = torch.randn(sample_shape, device=self.device)
            for step in range(self.num_steps - 1, -1, -1):
                noise = torch.randn_like(x)
                t = torch.full((B,), step, device=self.device, dtype=torch.long)

                if self.is_multimodal:
                    keep = torch.zeros(B, dtype=torch.bool, device=self.device)
                    drop = torch.ones(B, dtype=torch.bool, device=self.device)

                    attr_both = self.compute_condition_from_emb(text_emb, t, ref_images, keep, keep)
                    pred_both = self.dit(x, t, attr_both, intervention=intervention)

                    use_mm_cfg = (
                        guidance_scale != 1.0
                        or img_w != 1.0
                        or interaction_guidance_scale != 0.0
                    )
                    if use_mm_cfg:
                        attr_uncond = self.compute_condition_from_emb(text_emb, t, ref_images, drop, drop)
                        attr_text = self.compute_condition_from_emb(text_emb, t, ref_images, keep, drop)
                        attr_image = self.compute_condition_from_emb(text_emb, t, ref_images, drop, keep)
                        pred_uncond = self.dit(x, t, attr_uncond, intervention=intervention)
                        pred_text = self.dit(x, t, attr_text, intervention=intervention)
                        pred_image = self.dit(x, t, attr_image, intervention=intervention)
                        pred_noise = (
                            pred_uncond
                            + guidance_scale * (pred_text - pred_uncond)
                            + img_w * (pred_image - pred_uncond)
                            + interaction_guidance_scale * (pred_both - pred_text - pred_image + pred_uncond)
                        )
                    else:
                        pred_noise = pred_both
                else:
                    attr_emb = self.compute_condition_from_emb(text_emb, t)
                    pred_noise = self.dit(x, t, attr_emb, intervention=intervention)
                    if guidance_scale > 1.0 and texts is not None:
                        attr_uncond = self.compute_condition_from_emb(empty_text_emb, t)
                        pred_uncond = self.dit(x, t, attr_uncond, intervention=intervention)
                        pred_noise = pred_uncond + guidance_scale * (pred_noise - pred_uncond)

                if sampler == "ddpm":
                    x = self.ddpm.reverse(x, pred_noise, t, noise)
                else:
                    x = self.ddim.reverse(x, pred_noise, t, noise, is_determin=True)
            samples.append(x.clamp(0.0, 1.0))

        return torch.stack(samples)
