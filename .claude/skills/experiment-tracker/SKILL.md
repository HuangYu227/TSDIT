---
name: experiment-tracker
description: Track and manage MMLDM training experiments across branches. Find checkpoints, parse experiment naming conventions, and report training status.
---

# experiment-tracker

Track and manage MMLDM training experiments across branches with 4376+ .pt checkpoint files.

## When to Use

- User asks about experiment status, checkpoint locations, or training history
- User wants to find the best checkpoint for a given experiment
- User asks "what experiments are running" or "what was the config for X"
- User wants to compare experiments across branches

## Experiment Naming Convention

Experiment directories follow the pattern:
```
checkpoints/stage{N}_{branch}_{variant}/
```
Examples:
- `checkpoints/stage2_v2_cfg_textenc/` — Stage 2, V2 branch, cfg_textenc variant
- `checkpoints/stage1_t2s/` — Stage 1, T2S-style split
- `checkpoints/stage1_weather/` — Stage 1, Weather dataset
- `checkpoints/stage2_full/` — Stage 2, full config with all losses

Checkpoint files within each dir:
- `epoch_{N}.pt` — periodic saves
- `best.pt` — best validation loss
- `latest.pt` — most recent save

## Branch Map

| Branch | Purpose |
|--------|---------|
| `myverbal` | Active development — Weather dataset, VerbalTS eval, T2S split comparison |
| `feature/mmv2-spectral-dual-latent` | V2 pipeline with spectral dual-latent VAE + DiT |
| `mmldm-v4` | Clean training baseline (V2 codebase + warmup fix) |
| `main` | Stable/production |

## Config Reverse-Engineering

Given an experiment directory name, derive the likely training config:

**stage2_v2_cfg_textenc**: batch_size=512, lr=3e-4, dit_layers=8, dit_dim=256, dit_heads=4, block_size=8, cfg_drop_prob=0.3, batch_mul=1, gamma1=gamma2=gamma_cons=0. VAE text encoder unfrozen (trained by FM loss).

**stage2_full**: batch_mul=4, all auxiliary losses enabled (gamma1, gamma2, gamma_cons, gamma_freq > 0).

## Key Metrics

- **MSE** (normalized space): lower is better, <1.0 is excellent
- **WAPE** (raw space): lower is better, <2.0 is competitive with T2S (0.183)
- **CTTP**: cosine similarity between generated TS and text embeddings, higher is better
- **FID**: Frechet distance between train and gen TS embeddings, lower is better
- **JFTSD**: Joint FID including text embeddings, lower is better

## Commands

### Find all checkpoints
```bash
find checkpoints/ -name "*.pt" -type f | sort
```

### Show latest checkpoint per experiment
```bash
for d in checkpoints/*/; do echo "$d: $(ls -t "$d"*.pt 2>/dev/null | head -1)"; done
```

### Estimate experiment config from directory name
Parse the directory name tokens to reconstruct the likely CLI args used.
