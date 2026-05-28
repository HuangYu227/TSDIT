"""TIGER Evaluator.

Runs end-to-end evaluation: generate images from text descriptions,
convert back to TS, compute metrics (MSE, WAPE, MAE, MRE, PSNR, SSIM).
"""

import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data.dataset import TIGERDataset, TIGERCollateFn
from ..generator import TIGERGenerator
from ..image_to_ts import ImageToTSDecoder
from .metrics import (
    compute_all_ts_metrics,
    compute_psnr,
    compute_ssim,
    compute_roundtrip_error,
)


class TIGEREvaluator:
    """Evaluate TIGER on a test split.

    Parameters
    ----------
    config : dict
        Same config used for training (device, diffusion, condition, data).
    checkpoint_path : str
        Path to a saved .pth checkpoint.
    data_dir : str
        Dataset root directory.
    split : str
        Which split to evaluate ("test" or "valid").
    n_samples : int
        Number of diffusion samples per test input (median aggregated).
    sampler : str
        "ddim" or "ddpm".
    """

    def __init__(
        self,
        config: dict,
        checkpoint_path: str,
        data_dir: str,
        split: str = "test",
        n_samples: int = 10,
        sampler: str = "ddim",
    ):
        self.config = config
        self.device = config["device"]
        self.n_samples = n_samples
        self.sampler = sampler

        # Load model
        model_config = {
            "device": self.device,
            "diffusion": config["diffusion"],
            "condition": config["condition"],
        }
        self.model = TIGERGenerator(model_config)
        state = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()

        # Image -> TS decoder
        dc = config["data"]
        self.ts_decoder = ImageToTSDecoder(
            mode="fused",
            n_fft=dc.get("n_fft", 64),
            hop_length=dc.get("hop_length", 8),
        )

        # Test data
        self.dataset = TIGERDataset(
            data_dir=data_dir,
            split=split,
            dataset_type=dc["dataset_type"],
            image_size=dc.get("image_size", 64),
            n_fft=dc.get("n_fft", 64),
            hop_length=dc.get("hop_length", 8),
            epsilon_quantile=dc.get("epsilon_quantile", 0.1),
        )
        self.loader = DataLoader(
            self.dataset,
            batch_size=config.get("batch_size", 64),
            shuffle=False,
            collate_fn=TIGERCollateFn(),
            num_workers=0,
        )

    @torch.no_grad()
    def evaluate(self) -> dict:
        """Run full evaluation.

        Returns
        -------
        dict
            Keys: mse, mae, wape, mre, psnr, ssim, roundtrip_mse, etc.
        """
        all_pred_ts = []
        all_true_ts = []
        all_gen_images = []
        all_real_images = []

        dc = self.config["data"]
        image_size = dc.get("image_size", 64)
        image_shape = (3, image_size, image_size)

        for batch_idx, batch in enumerate(self.loader):
            images = batch["image"].to(self.device).float()
            texts = batch.get("cap", None)
            ts_true = batch["ts"].float()  # (B, T) normalized
            ts_len = batch["ts_len"]
            ts_min = batch["ts_min"]
            ts_max = batch["ts_max"]

            # Generate n_samples images per input (text-only conditioning)
            gen_images = self.model.generate(
                image_shape, texts, n_samples=self.n_samples, sampler=self.sampler
            )  # (n_samples, B, 3, H, W)

            # Take median over samples
            gen_median = gen_images.median(dim=0).values  # (B, 3, H, W)

            # Convert generated images back to TS
            from ..ts_to_image import NormParams
            norm_params = NormParams(
                min_val=ts_min,
                max_val=ts_max,
                n_vars=1,
                original_length=ts_len,
            )
            pred_ts = self.ts_decoder.decode(
                gen_median, ts_length=ts_len, norm_params=norm_params
            )  # (B, T)

            # Denormalize ground truth to same scale
            true_ts = ts_true * (ts_max - ts_min).unsqueeze(-1) + ts_min.unsqueeze(-1)

            all_pred_ts.append(pred_ts.cpu().numpy())
            all_true_ts.append(true_ts.cpu().numpy())
            all_gen_images.append(gen_median.cpu().numpy())
            all_real_images.append(images.cpu().numpy())

            if (batch_idx + 1) % 10 == 0:
                print(f"  Evaluated batch {batch_idx + 1}")

        # Concatenate
        pred_ts = np.concatenate(all_pred_ts, axis=0)
        true_ts = np.concatenate(all_true_ts, axis=0)
        gen_images = np.concatenate(all_gen_images, axis=0)
        real_images = np.concatenate(all_real_images, axis=0)

        # Compute metrics
        results = compute_all_ts_metrics(pred_ts, true_ts)
        results["psnr"] = compute_psnr(gen_images, real_images)
        results["ssim"] = compute_ssim(gen_images, real_images)

        # Round-trip error (TS -> Image -> TS on ground truth)
        rt_images, rt_norm = self._encode_ts_batch(true_ts, ts_len, ts_min, ts_max)
        rt_ts = self.ts_decoder.decode(
            torch.tensor(rt_images, device=self.device).float(),
            ts_length=ts_len,
            norm_params=rt_norm.to(self.device),
        )
        rt_metrics = compute_roundtrip_error(
            torch.tensor(true_ts), rt_ts.cpu()
        )
        results.update(rt_metrics)

        return results

    def _encode_ts_batch(self, ts_np, ts_len, ts_min, ts_max):
        """Encode a batch of TS to images for round-trip testing."""
        from ..ts_to_image import TSToImageEncoder, NormParams

        encoder = TSToImageEncoder(
            image_size=self.config["data"].get("image_size", 64),
            n_fft=self.config["data"].get("n_fft", 64),
            hop_length=self.config["data"].get("hop_length", 8),
        )
        ts_tensor = torch.tensor(ts_np, dtype=torch.float32)
        ts_min_t = ts_tensor.min(dim=-1, keepdim=True).values
        ts_max_t = ts_tensor.max(dim=-1, keepdim=True).values
        ts_range = (ts_max_t - ts_min_t).clamp(min=1e-8)
        ts_norm = (ts_tensor - ts_min_t) / ts_range

        images, _ = encoder.encode(ts_norm)
        norm_params = NormParams(
            min_val=ts_min_t.squeeze(-1),
            max_val=ts_max_t.squeeze(-1),
            n_vars=1,
            original_length=ts_len,
        )
        return images.numpy(), norm_params

    def print_results(self, results: dict):
        """Pretty-print evaluation results."""
        print("\n" + "=" * 50)
        print("TIGER Evaluation Results")
        print("=" * 50)
        print(f"  MSE:            {results['mse']:.6f}")
        print(f"  MAE:            {results['mae']:.6f}")
        print(f"  WAPE:           {results['wape']:.6f}")
        print(f"  MRE:            {results['mre']:.6f}")
        print(f"  PSNR:           {results['psnr']:.2f} dB")
        print(f"  SSIM:           {results['ssim']:.4f}")
        print(f"  Round-trip MSE: {results['roundtrip_mse']:.6f}")
        print(f"  Round-trip MAE: {results['roundtrip_mae']:.6f}")
        print("=" * 50)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    import argparse
    import json
    from ..train import load_config, set_seed

    parser = argparse.ArgumentParser(description="TIGER Evaluation")
    parser.add_argument("--config", type=str, required=True, help="Config JSON (from training)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint .pth")
    parser.add_argument("--data_dir", type=str, required=True, help="Dataset directory")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--sampler", type=str, default="ddim", choices=["ddim", "ddpm"])
    args = parser.parse_args()

    config = load_config(args.config)
    if torch.cuda.is_available():
        config["device"] = "cuda:0"
    else:
        config["device"] = "cpu"

    set_seed(config.get("seed", 42))

    evaluator = TIGEREvaluator(
        config=config,
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        split=args.split,
        n_samples=args.n_samples,
        sampler=args.sampler,
    )

    results = evaluator.evaluate()
    evaluator.print_results(results)

    save_path = os.path.join(
        os.path.dirname(args.checkpoint), "eval_results.json"
    )
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {save_path}")


if __name__ == "__main__":
    main()
