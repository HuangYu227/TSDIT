import torch
import torch.nn as nn
from .dit_model import TIGERDiT
from .cond_projector import TextOnlyProjector
from .samplers import DDPMSampler, DDIMSampler


class TIGERGenerator(nn.Module):
    """Text-conditioned diffusion generator for time-series-to-image.

    Flow: noisy image + text + diffusion_step -> predicted noise
    Conditioning is text-only; no image encoder is used.
    """

    def __init__(self, config):
        super().__init__()
        self.device = config["device"]
        self.config = config

        diff_config = config["diffusion"]
        cond_config = config["condition"]

        self._init_text_encoder(cond_config)
        self._init_cond_projector(diff_config, cond_config)
        self._init_dit(diff_config)

        self.num_steps = diff_config["num_steps"]
        self.ddpm = DDPMSampler(
            self.num_steps, diff_config["beta_start"], diff_config["beta_end"],
            diff_config.get("schedule", "quad"), self.device
        )
        self.ddim = DDIMSampler(
            self.num_steps, diff_config["beta_start"], diff_config["beta_end"],
            diff_config.get("schedule", "quad"), self.device
        )

    def _init_text_encoder(self, cond_config):
        from transformers import AutoTokenizer, CLIPTextModelWithProjection, CLIPTextConfig
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
            nn.Linear(cond_config["text"]["textemb_hidden_dim"], cond_config["text"]["text_emb"])
        ).to(self.device)

    def _init_cond_projector(self, diff_config, cond_config):
        n_var = diff_config.get("n_var", 16)
        n_scale = diff_config.get("multipatch_num", 1)
        n_steps = diff_config["num_steps"]
        n_stages = cond_config.get("num_stages", 4)

        self.cond_projector = TextOnlyProjector(
            n_var=n_var, n_scale=n_scale, n_steps=n_steps, n_stages=n_stages,
            dim_in_text=cond_config["text"]["text_emb"],
            dim_out=diff_config["channels"]
        ).to(self.device)

    def _init_dit(self, diff_config):
        self.dit = TIGERDiT(diff_config).to(self.device)

    def encode_text(self, texts):
        """Encode text descriptions to token-level embeddings.

        Returns (B, N_tokens, text_emb) — a sequence, matching image encoder output.
        """
        max_len = self.text_model.config.max_position_embeddings
        inputs = self.text_tokenizer(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        # last_hidden_state: (B, seq_len, hidden_dim) — all token embeddings
        text_hidden = self.text_model(**inputs).last_hidden_state
        text_emb = self.text_proj(text_hidden)  # (B, seq_len, text_emb)
        return text_emb

    def compute_condition(self, texts, batch_size, diffusion_step):
        """Compute condition embedding from text only.

        Args:
            texts: list of text descriptions, or None for unconditional.
            batch_size: batch size (needed when texts is None for CFG).
            diffusion_step: (B,) integer timestep tensor.
        """
        if texts is not None:
            text_emb = self.encode_text(texts)
        else:
            text_emb = self.encode_text([""] * batch_size)
        return self.cond_projector(text_emb, diffusion_step)

    def _noise_estimation_loss(self, x, attr_emb, t):
        """Compute noise estimation loss for a given timestep."""
        noise = torch.randn_like(x)
        noisy_x = self.ddpm.forward(x, t, noise)

        pred_noise = self.dit(noisy_x, t, attr_emb)
        residual = noise - pred_noise
        noise_loss = (residual ** 2).mean()

        loss = noise_loss
        loss_dict = {
            "noise_loss": noise_loss.detach(),
        }

        # --- CTICD: add causal auxiliary losses if available ---
        if hasattr(self.dit, '_cticd_losses') and self.dit._cticd_losses is not None:
            lambda_cticd = self.config["diffusion"].get("lambda_cticd", 0.1)
            cticd_total = self.dit._cticd_losses['cticd_total']
            loss = loss + lambda_cticd * cticd_total
            loss_dict["cticd_weighted"] = (lambda_cticd * cticd_total).detach()
            for k, v in self.dit._cticd_losses.items():
                loss_dict[k] = v.detach()

        # --- CSA-MoE: add auxiliary load-balancing loss if available ---
        if hasattr(self.dit, '_csa_moe_losses') and self.dit._csa_moe_losses is not None:
            lambda_moe = self.config["diffusion"].get("lambda_moe", 0.1)
            loss = loss + lambda_moe * self.dit._csa_moe_losses
            loss_dict["moe_aux"] = self.dit._csa_moe_losses.detach()

        loss_dict["all"] = loss
        return loss_dict

    def forward(self, batch, is_train=True):
        """Training forward pass."""
        images = batch["image"].to(self.device).float()
        texts = batch.get("cap", None)
        B = images.shape[0]

        if is_train:
            t = torch.randint(0, self.num_steps, [B], device=self.device)

            # CFG: randomly drop text conditioning during training
            cfg_dropout = self.config["condition"].get("cfg_dropout", 0.0)
            if cfg_dropout > 0 and texts is not None:
                if torch.rand(1).item() < cfg_dropout:
                    texts = None

            attr_emb = self.compute_condition(texts, B, t)
            return self._noise_estimation_loss(images, attr_emb, t)

        loss_acc = {}
        for step in range(self.num_steps):
            t = torch.full((B,), step, device=self.device, dtype=torch.long)
            attr_emb = self.compute_condition(texts, B, t)
            step_losses = self._noise_estimation_loss(images, attr_emb, t)
            for k, v in step_losses.items():
                loss_acc[k] = loss_acc.get(k, 0.0) + v.item()

        return {k: v / self.num_steps for k, v in loss_acc.items()}

    @torch.no_grad()
    def generate(self, image_shape, texts, n_samples=1, sampler="ddim",
                 guidance_scale: float = 1.0):
        """Generate images conditioned on text only.

        Args:
            image_shape: tuple (C, H, W) for generated image dimensions.
            texts: list of text descriptions.
            n_samples: number of samples per text.
            sampler: "ddim" or "ddpm".
            guidance_scale: CFG guidance weight. 1.0 = no guidance,
                >1.0 applies classifier-free guidance.

        Returns:
            Tensor of shape (n_samples, B, C, H, W).
        """
        B = len(texts) if texts is not None else 1
        sample_shape = (B, *image_shape)
        samples = []
        use_cfg = guidance_scale > 1.0 and texts is not None

        for i in range(n_samples):
            x = torch.randn(sample_shape, device=self.device)
            for step in range(self.num_steps - 1, -1, -1):
                noise = torch.randn_like(x)
                t = torch.full((B,), step, device=self.device, dtype=torch.long)
                attr_emb = self.compute_condition(texts, B, t)
                pred_noise = self.dit(x, t, attr_emb)

                # CFG: blend conditional and unconditional predictions
                if use_cfg:
                    attr_uncond = self.compute_condition(None, B, t)
                    pred_uncond = self.dit(x, t, attr_uncond)
                    pred_noise = pred_uncond + guidance_scale * (pred_noise - pred_uncond)

                if sampler == "ddpm":
                    x = self.ddpm.reverse(x, pred_noise, t, noise)
                else:
                    x = self.ddim.reverse(x, pred_noise, t, noise, is_determin=True)
            samples.append(x)

        return torch.stack(samples)
