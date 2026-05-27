import torch
import torch.nn as nn
from .dit_model import TIGERDiT
from .image_encoder import ImageEncoder, ViTEncoder, CNNEncoder
from .cond_projector import ImageTextProjector, ImageOnlyProjector, TextOnlyProjector
from .samplers import DDPMSampler, DDIMSampler


class TIGERGenerator(nn.Module):
    """Main TIGER generator: Text+Image conditioned diffusion on images.
    
    Adapted from VerbalTS ConditionalGenerator.
    Flow: noisy image + text + diffusion_step -> predicted noise
    """

    def __init__(self, config):
        super().__init__()
        self.device = config["device"]
        self.config = config
        
        diff_config = config["diffusion"]
        cond_config = config["condition"]

        self._init_image_encoder(cond_config)
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

    def _init_image_encoder(self, cond_config):
        encoder_type = cond_config.get("image_encoder_type", "cnn")
        if encoder_type == "clip":
            self.image_encoder = ImageEncoder(cond_config["image"]).to(self.device)
        elif encoder_type == "vit":
            self.image_encoder = ViTEncoder(cond_config["image"]).to(self.device)
        else:
            self.image_encoder = CNNEncoder(cond_config["image"]).to(self.device)

    def _init_text_encoder(self, cond_config):
        if cond_config.get("use_text", True):
            from transformers import AutoTokenizer, CLIPTextModelWithProjection
            model_path = cond_config["text"].get("pretrain_model_path", "openai/clip-vit-base-patch32")
            self.text_tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.text_model = CLIPTextModelWithProjection.from_pretrained(model_path).to(self.device)
            for param in self.text_model.parameters():
                param.requires_grad = False
            self.text_proj = nn.Sequential(
                nn.Linear(cond_config["text"]["pretrain_model_dim"], cond_config["text"]["textemb_hidden_dim"]),
                nn.LayerNorm(cond_config["text"]["textemb_hidden_dim"]),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(cond_config["text"]["textemb_hidden_dim"], cond_config["text"]["text_emb"])
            ).to(self.device)
        else:
            self.text_tokenizer = None
            self.text_model = None
            self.text_proj = None

    def _init_cond_projector(self, diff_config, cond_config):
        n_var = diff_config.get("n_var", 16)
        n_scale = diff_config.get("multipatch_num", 1)
        n_steps = diff_config["num_steps"]
        n_stages = cond_config.get("num_stages", 4)
        cond_mode = cond_config.get("cond_mode", "text+image")

        if cond_mode == "text+image":
            self.cond_projector = ImageTextProjector(
                n_var=n_var, n_scale=n_scale, n_steps=n_steps, n_stages=n_stages,
                dim_in_text=cond_config["text"]["text_emb"],
                dim_in_image=cond_config["image"]["image_emb"],
                dim_out=diff_config["channels"]
            ).to(self.device)
        elif cond_mode == "text_only":
            self.cond_projector = TextOnlyProjector(
                n_var=n_var, n_scale=n_scale, n_steps=n_steps, n_stages=n_stages,
                dim_in_text=cond_config["text"]["text_emb"],
                dim_out=diff_config["channels"]
            ).to(self.device)
        elif cond_mode == "image_only":
            self.cond_projector = ImageOnlyProjector(
                n_var=n_var, n_scale=n_scale, n_steps=n_steps, n_stages=n_stages,
                dim_in_image=cond_config["image"]["image_emb"],
                dim_out=diff_config["channels"]
            ).to(self.device)
        else:
            raise ValueError(f"Unknown cond_mode: {cond_mode}")

    def _init_dit(self, diff_config):
        self.dit = TIGERDiT(diff_config).to(self.device)

    def encode_text(self, texts):
        """Encode text descriptions to token-level embeddings.

        Returns (B, N_tokens, text_emb) — a sequence, matching image encoder output.
        """
        if self.text_model is None:
            return None
        inputs = self.text_tokenizer(texts, padding=True, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        # last_hidden_state: (B, seq_len, hidden_dim) — all token embeddings
        text_hidden = self.text_model(**inputs).last_hidden_state
        text_emb = self.text_proj(text_hidden)  # (B, seq_len, text_emb)
        return text_emb

    def encode_image(self, images):
        """Encode conditioning images to embeddings."""
        return self.image_encoder(images)

    def compute_condition(self, images, texts, diffusion_step):
        """Compute condition embedding based on cond_mode."""
        cond_mode = self.config["condition"].get("cond_mode", "text+image")

        if cond_mode == "text+image":
            image_emb = self.encode_image(images)
            # For CFG: texts=None → encode empty string as null-text embedding
            if texts is not None:
                text_emb = self.encode_text(texts)
            else:
                text_emb = self.encode_text([""] * images.shape[0])
            attr_emb = self.cond_projector(text_emb, image_emb, diffusion_step)
        elif cond_mode == "text_only":
            if texts is not None:
                text_emb = self.encode_text(texts)
            else:
                text_emb = self.encode_text([""] * (images.shape[0] if images is not None else 1))
            attr_emb = self.cond_projector(text_emb, diffusion_step)
        elif cond_mode == "image_only":
            image_emb = self.encode_image(images)
            attr_emb = self.cond_projector(image_emb, diffusion_step)
        else:
            raise ValueError(f"Unknown cond_mode: {cond_mode}")

        return attr_emb

    def _noise_estimation_loss(self, x, attr_emb, t):
        """Compute noise estimation loss for a given timestep."""
        noise = torch.randn_like(x)
        noisy_x = self.ddpm.forward(x, t, noise)
        
        pred_noise = self.dit(noisy_x, t, attr_emb)
        residual = noise - pred_noise
        return (residual ** 2).mean()

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

            attr_emb = self.compute_condition(images, texts, t)
            loss = self._noise_estimation_loss(images, attr_emb, t)
            return {"noise_loss": loss, "all": loss}

        loss_total = 0.0
        for step in range(self.num_steps):
            t = torch.full((B,), step, device=self.device, dtype=torch.long)
            attr_emb = self.compute_condition(images, texts, t)
            loss_total += self._noise_estimation_loss(images, attr_emb, t)

        avg_loss = loss_total / self.num_steps
        return {"noise_loss": avg_loss, "all": avg_loss}

    @torch.no_grad()
    def generate(self, images, texts, n_samples=1, sampler="ddim",
                 guidance_scale: float = 1.0):
        """Generate images conditioned on text + reference image.

        Args:
            guidance_scale: CFG guidance weight. 1.0 = no guidance,
                >1.0 applies classifier-free guidance.
        """
        B = images.shape[0]
        sample_shape = images.shape
        samples = []
        use_cfg = guidance_scale > 1.0 and texts is not None

        for i in range(n_samples):
            x = torch.randn(sample_shape, device=self.device)
            for step in range(self.num_steps - 1, -1, -1):
                noise = torch.randn_like(x)
                t = torch.full((B,), step, device=self.device, dtype=torch.long)
                attr_emb = self.compute_condition(images, texts, t)
                pred_noise = self.dit(x, t, attr_emb)

                # CFG: blend conditional and unconditional predictions
                if use_cfg:
                    attr_uncond = self.compute_condition(images, None, t)
                    pred_uncond = self.dit(x, t, attr_uncond)
                    pred_noise = pred_uncond + guidance_scale * (pred_noise - pred_uncond)

                if sampler == "ddpm":
                    x = self.ddpm.reverse(x, pred_noise, t, noise)
                else:
                    x = self.ddim.reverse(x, pred_noise, t, noise, is_determin=True)
            samples.append(x)

        return torch.stack(samples)
