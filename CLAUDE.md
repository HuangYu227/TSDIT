# MMLDM — Multimodal Latent Diffusion Model for Text-to-Time-Series

V2 architecture: Spectral Dual-Latent VAE → DiT Flow Matching with Text-Guided Feature Modulation.

## Branch Conventions

| Branch | Purpose |
|--------|---------|
| `myverbal` | **Active** — Weather dataset, VerbalTS eval, T2S split comparison |
| `feature/mmv2-spectral-dual-latent` | V2 full pipeline (Stage 1+2, all innovations) |
| `mmldm-v4` | Clean training baseline (V2 codebase + warmup fix) |
| `main` | Stable, do not commit directly |

Work on `myverbal` by default. Merge to `feature/mmv2` for stable V2 releases.

## Training Command Templates

### Stage 1 — VAE (Spectral Dual-Latent)

```bash
CUDA_VISIBLE_DEVICES=0 python -m mmldm.training_stage1 \
    --data_dir "./Three Levels Data/TSFragment-600K" \
    --split_file ./data/splits_t2s.json \
    --datasets ETTh1 --time_intervals 24 \
    --epochs 200 --batch_size 512 --lr 1e-4 \
    --dim 256 --latent_dim 64 \
    --num_conv_layers 4 --encoder_blocks 6 --decoder_blocks 6 \
    --kl_anneal_epochs 10 --kl_anneal_end 1e-5 \
    --gamma_spectral 0.1 --gamma_tclr 0.1 \
    --save_dir ./checkpoints/stage1_<name> --seed 42
```

### Stage 2 — DiT Flow Matching (cfg_textenc — best config)

```bash
CUDA_VISIBLE_DEVICES=0 python -m mmldm.training_stage2 \
    --data_dir "./Three Levels Data/TSFragment-600K" \
    --vae_checkpoint <vae_ckpt> \
    --split_file <split_json> \
    --datasets ETTh1 --time_intervals 24 \
    --epochs 500 --batch_size 512 --lr 3e-4 --warmup_steps 500 \
    --gamma1 0.0 --gamma2 0.0 \
    --dit_dim 256 --dit_layers 8 --dit_heads 4 --block_size 8 \
    --cfg_drop_prob 0.3 --gamma_cons 0.0 --batch_mul 1 \
    --log_interval 10 --save_dir <save_dir> --seed 42
```

Key finding: `batch_mul=1` and all extra losses disabled (`gamma1=gamma2=gamma_cons=0`) significantly outperforms the full config.

### Inference

```bash
python -m mmldm.inference \
    --vae_checkpoint <vae_ckpt> --dit_checkpoint <dit_ckpt> \
    --split_file <split_json> --datasets ETTh1 --time_intervals 24 \
    --n_runs 10 --guidance_scale 7.0 --batch_size 64 \
    --save_path ./outputs/generated.npy
```

## Data Splits

- **T2S-style** (99/1, seed=2023): `./data/splits_t2s.json` — use for fair comparison
- **Original** (80/10/10, seed=42): use for development

## Memory System

Project memory lives at `~/.claude/projects/E--Research-TSG-myTSG-V0/memory/MEMORY.md`. Key entries:
- [[stage2-cfg-textenc-best]] — best Stage 2 config details
- [[t2s-split-difference]] — T2S split vs our original split

## CodeGraph

This project has CodeGraph MCP indexed. Use `codegraph_*` tools for structural queries (where is X defined, what calls Y, what would changing Z break). Prefer codegraph over grep for code structure questions.

## Key Metrics

- **MSE** (normalized): <1.0 excellent, cfg_textenc=2.98 (early stop)
- **WAPE** (raw): <2.0 competitive, cfg_textenc=1.63, T2S=0.183
- **CTTP/FID/JFTSD**: VerbalTS metrics, computed via `eval_verbalts.py`
