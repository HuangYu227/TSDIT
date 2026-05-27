# MMLDM V2 架构图 (feature/mmv2-spectral-dual-latent)

## 整体两阶段训练

```mermaid
graph TB
    subgraph "数据管线"
        TS[时序数据<br/>ETTh1 × 24h]
        SBERT[SBERT 文本嵌入<br/>128-dim]
        TS --> |TSFragmentDataset| FRAG[时序片段<br/>(L, C) per sample]
        SBERT --> EMB[文本嵌入<br/>(B, 128)]
    end

    subgraph "Stage 1: VAE 训练"
        FRAG --> VAE[MMLDMVAEModel<br/>Spectral Dual-Latent VAE]
        FFT[FFT 分解<br/>cutoff_ratio=0.3]
        FFT --> |低频趋势| TREND_ENC[trend_encoder<br/>Conv1d × ConvResidualStack]
        FFT --> |高频残差| RESID_ENC[residual_encoder<br/>Conv1d × ConvResidualStack]
        TREND_ENC --> TREND_PROJ[trend_proj → z_trend<br/>DiagonalGaussian]
        RESID_ENC --> RESID_PROJ[residual_proj → z_residual<br/>DiagonalGaussian]
        TREND_PROJ --> MERGE[[z = z_trend; z_residual]]
        RESID_PROJ --> MERGE
        MERGE --> STD[潜变量标准化<br/>dataset-level μ, σ]
        STD --> DECODER[Conv1d Decoder → x_recon]
        DECODER --> L_RECON[MSE Loss<br/>L_recon]
        TREND_PROJ --> KL[KL Loss<br/>L_kl]
        RESID_PROJ --> KL
        DECODER --> SPECTRAL[频谱重建 Loss<br/>per-sample FFT L1]
        MERGE --> TCLR[TCLR Loss<br/>时序对比正则化]
        L_RECON --> L1_TOTAL[Stage1 Total Loss]
        KL --> L1_TOTAL
        SPECTRAL --> L1_TOTAL
        TCLR --> L1_TOTAL
    end

    subgraph "从 Stage1 到 Stage2"
        STD --> LATENT_STATS[latent_mean & latent_std<br/>存入 checkpoint]
        MERGE --> Z0[潜变量 z₀<br/>(L, latent_dim=64)]
    end

    subgraph "Stage 2: DiT 训练"
        Z0 --> DIT[MMLDMDiTModel<br/>Flow Matching]
        EMB --> |text_raw B×128| MVTC
        EMB --> |text_emb| VAE_ENC[VAE.encode_text_condition]
        VAE_ENC --> |B×64| DIT
    end

    style VAE fill:#e1f5fe
    style DIT fill:#fff3e0
    style FFT fill:#f3e5f5
    style MVTC fill:#e8f5e9
```

## VAE 架构细节 (modeling_mmldm_vae.py)

```mermaid
graph LR
    subgraph "编码器 (encode)"
        X[输入时序<br/>ot_list] --> UNSQ[unsqueeze+permute<br/>B×C×L → 1×C×L]
        UNSQ --> FFT_DECOMP[fft_decompose<br/>rfft → 截断 → irfft]
        FFT_DECOMP --> XLOW[x_low 低频趋势]
        FFT_DECOMP --> XHIGH[x_high 高频残差]
        XLOW --> TE[trend_encoder<br/>Conv1d k3p1 → SiLU →<br/>ConvResidualStack × N]
        XHIGH --> RE[residual_encoder<br/>Conv1d k3p1 → SiLU →<br/>ConvResidualStack × N]
        TE --> TP[trend_proj<br/>Conv1d k1 → μ+logvar]
        RE --> RP[residual_proj<br/>Conv1d k1 → μ+logvar]
        TP --> TD[DiagonalGaussian<br/>trend_dist]
        RP --> RD[DiagonalGaussian<br/>residual_dist]
    end

    subgraph "解码器 (decode)"
        Z[z = z_trend; z_residual<br/>L × 64] --> UNSTD[unstandardize_latent<br/>z × σ + μ]
        UNSTD --> DI[decoder_in_layer<br/>Conv1d k1 → dim]
        DI --> DR[ConvResidualStack × N]
        DR --> DO[proj_out<br/>Conv1d k1 → C]
        DO --> RECON[x_recon]
    end

    subgraph "文本编码器"
        TE2[text_embs<br/>B × 128] --> TPROJ[text_proj<br/>Linear → dim]
        TPROJ --> TXBLOCK[TextTransformerBlock × N<br/>+ diagonal attention mask]
        TXBLOCK --> TFN[text_final_norm + Linear<br/>→ latent_dim=64]
        TFN --> TLAT[text_latent]
    end

    subgraph "Latent 标准化"
        ALLZ[所有训练样本的 z] --> COMPUTE[compute_latent_stats<br/>μ = mean, σ = std]
        COMPUTE --> BUFFER["register_buffer<br/>latent_mean, latent_std<br/>persistent=True"]
    end

    style FFT_DECOMP fill:#f3e5f5
    style TE fill:#e8eaf6
    style RE fill:#e8eaf6
    style TXBLOCK fill:#fff9c4
```

## DiT 架构细节 (modeling_mmldm_dit.py)

```mermaid
graph TB
    subgraph "输入"
        ZT[zₜ 加噪潜变量<br/>L_ts × 64]
        TLAT[text_latent<br/>B × 64]
        TRAW[text_raw SBERT<br/>B × 128]
        TSTEP[timestep t]
    end

    subgraph "Patch 嵌入"
        ZT --> TSPATCH[PatchIn1D<br/>ts_in: 64 → txt_dim]
        TLAT --> TXPATCH[PatchIn1D<br/>text_in: 64 → txt_dim]
        TSTEP --> TEMB[TimestepEmbedding<br/>sinusoidal → emb_dim]
    end

    subgraph "MVTC 文本多视图"
        TRAW --> TEXT_POOLER[MultiViewTextPooler<br/>K=4 正交线性投影]
        TEXT_POOLER --> V0[view_0<br/>B×128]
        TEXT_POOLER --> V1[view_1<br/>B×128]
        TEXT_POOLER --> V2[view_2<br/>B×128]
        TEXT_POOLER --> V3[view_3<br/>B×128]
    end

    subgraph "DiT Blocks × 12"
        TSPATCH --> B0[Block 0 ← view_0]
        TXPATCH --> B0
        TEMB --> B0
        B0 --> B1[Block 1 ← view_1]
        B1 --> B2[Block 2 ← view_2]
        B2 --> B3[Block 3 ← view_3]
        B3 --> B4[Block 4 ← view_0]
        B4 --> BDOTS[...]
        BDOTS --> B11[Block 11 ← view_3]
    end

    subgraph "MultimodalDiTBlock 内部"
        direction LR
        subgraph "TS 路径"
            TSIN[ts tokens] --> ADALN1[adaLN<br/>t_emb → γ,β,α]
            ADALN1 --> ATTN1[MultimodalJointAttention<br/>block-causal mask]
            ATTN1 --> |+ residual| TSFFN[FFN + adaLN]
            TSFFN --> TGFM[TextModulator<br/>scale, shift ← text_latent<br/>γ_text × gate]
            TGFM --> |+ residual| TSOUT[ts tokens']
        end
        subgraph "Text 路径"
            TXIN[text tokens] --> ADALN2[adaLN<br/>t_emb → γ,β,α]
            ADALN2 --> ATTN2[MultimodalJointAttention<br/>ts+text 联合注意力]
            ATTN2 --> |+ residual| TXFFN[FFN + adaLN]
            TXFFN --> TXOUT[text tokens']
        end
    end

    subgraph "输出"
        B11 --> ONORM[ts_out_norm + text_out_norm<br/>adaLN conditioned]
        ONORM --> OUTPROJ[ts_out.proj + text_out.proj<br/>zero-init]
        OUTPROJ --> VPRED[v_pred 速度场预测]
    end

    style TEXT_POOLER fill:#e8f5e9
    style TGFM fill:#fff3e0
    style ATTN1 fill:#e3f2fd
    style ATTN2 fill:#e3f2fd
```

## 训练流程

```mermaid
flowchart TD
    subgraph "Stage 1 训练 (training_stage1.py)"
        S1_INIT[初始化 VAE<br/>MMLDMVAEModel] --> S1_EPOCH{epoch loop}
        S1_EPOCH --> KL_BETA[KL β annealing<br/>start→end over N epochs]
        KL_BETA --> S1_BATCH{遍历 batch}
        S1_BATCH --> S1_FWD[model.forward<br/>encode → sample → standardize → decode]
        S1_FWD --> S1_LOSS["L = L_recon + β·KL<br/>+ γ_spec·L_spectral + γ_tclr·L_TCLR"]
        S1_LOSS --> S1_BACK[backward + clip_grad + step]
        S1_BACK --> |epoch==0| LATENT_STATS[compute_latent_stats<br/>存入 buffer]
        S1_BACK --> S1_CKPT[save checkpoint .pt]
    end

    subgraph "Stage 2 训练 (training_stage2.py)"
        S2_LOAD[加载 VAE checkpoint<br/>+ latent stats] --> S2_INIT[初始化 DiT<br/>MMLDMDiTModel]
        S2_INIT --> S2_EPOCH{epoch loop<br/>OneCycleLR}
        S2_EPOCH --> S2_SAMPLE[采样 z₀ = VAE.encode<br/>标准化后的潜变量]
        S2_SAMPLE --> S2_BATCHMUL[batch_mul<br/>z₀ repeat ×4]
        S2_BATCHMUL --> S2_TIMESTEP[采样 t ~ U0,1<br/>text_adaptive SNR]
        S2_TIMESTEP --> S2_CFG[CFG dropout 0.3<br/>√ text_latent × mask<br/>√ text_emb × mask]
        S2_CFG --> S2_FWD[DiT.forward<br/>zₜ = t·noise + 1-t·z₀<br/>v_pred = DiT zₜ, t, text, text_latent]
        S2_FWD --> S2_LOSS["L = L_FM<br/>+ γ1·L_DCD_mix<br/>+ γ2·L_DCD_aux<br/>+ γ_cons·L_consistency"]
        S2_LOSS --> S2_BACK[backward + step<br/>EMA update 0.9999]
    end

    style S1_LOSS fill:#e1f5fe
    style S2_LOSS fill:#fff3e0
```

## 推理流程 (inference.py)

```mermaid
flowchart TD
    INF_START[输入: 文本查询 + 条件] --> INF_SBERT[SBERT 编码<br/>→ text_emb B×128]
    INF_SBERT --> INF_VAE_ENC[VAE.encode_text_condition<br/>→ text_latent B×64]
    INF_VAE_ENC --> INF_ROUTER[SemanticRouter<br/>block 分配 + alignment]
    INF_ROUTER --> INF_INIT[初始化 z_T ~ N0,1<br/>标准化空间]
    INF_INIT --> INF_LOOP{timestep loop<br/>t = 1→0}
    INF_LOOP --> INF_CFG{"CFG?<br/>guidance_scale>1"}
    INF_CFG -->|是| INF_COND[v_cond = DiT zₜ, t, text<br/>text_latent=text_emb]
    INF_COND --> INF_UNCOND[v_uncond = DiT zₜ, t, ∅<br/>text_latent=0]
    INF_UNCOND --> INF_COMBINE["v = v_uncond<br/>+ w·(v_cond-v_uncond)"]
    INF_CFG -->|否| INF_DIRECT[v = DiT zₜ, t, text<br/>text_latent=text_emb]
    INF_DIRECT --> INF_EULER["Euler ODE step<br/>z_{t-Δt} = zₜ - Δt·v"]
    INF_COMBINE --> INF_EULER
    INF_EULER --> |KV Cache<br/>复用| INF_KV[prefix_kv / extend_prefix_kv<br/>block-causal 增量生成]
    INF_KV --> INF_LOOP
    INF_LOOP --> INF_FINAL[z₀ 最终潜变量]
    INF_FINAL --> INF_UNSTD[unstandardize_latent<br/>z₀ × σ + μ]
    INF_UNSTD --> INF_DECODE[VAE.decode<br/>z₀ → x̂]
    INF_DECODE --> INF_OUT["输出: 生成时序 x̂<br/>+ 评估指标 MSE,WAPE,MRR"]

    style INF_EULER fill:#e8f5e9
    style INF_CFG fill:#fff9c4
    style INF_DECODE fill:#e1f5fe
```

## 模块依赖关系

```mermaid
graph TB
    subgraph "支撑层"
        CONFIG[configuration_mmldm.py<br/>VAEConfig + DiTConfig]
        ATTN[attention_utils.py<br/>block-causal mask<br/>RoPE position encoding]
        ROUTER[semantic_router.py<br/>BoundaryDetector<br/>BlockAllocator]
    end

    subgraph "数据层"
        DS[data/tsfragment_dataset.py<br/>TSFragmentDataset<br/>CollateFn]
        SPLIT[data/split_dataset.py<br/>Train/Val 切分]
    end

    subgraph "模型层"
        VAE_MODEL[modeling_mmldm_vae.py<br/>Spectral Dual-Latent VAE<br/>+ TextTransformerBlock<br/>+ TCLR + LatentStd]
        DIT_MODEL[modeling_mmldm_dit.py<br/>MMDiT with block-causal<br/>+ TGFM TextModulator<br/>+ MVTC MultiViewTextPooler<br/>+ DCD denoising]
    end

    subgraph "训练层"
        S1_TRAIN[training_stage1.py<br/>VAE 训练: recon+KL+spectral+TCLR<br/>+ KL annealing + latent stats]
        S2_TRAIN[training_stage2.py<br/>DiT 训练: FM+DCD+Consistency<br/>+ OneCycleLR + EMA + CFG<br/>+ FreqLoss + BatchMul]
    end

    subgraph "推理层"
        INF[inference.py<br/>Euler ODE + CFG<br/>+ SemanticRouting<br/>+ block-causal generation]
        EVAL[evaluation.py<br/>MSE, WAPE, MRR]
    end

    CONFIG --> VAE_MODEL
    CONFIG --> DIT_MODEL
    ATTN --> DIT_MODEL
    ATTN --> ROUTER
    DS --> S1_TRAIN
    DS --> S2_TRAIN
    DS --> INF
    SPLIT --> S1_TRAIN
    SPLIT --> S2_TRAIN
    VAE_MODEL --> S1_TRAIN
    VAE_MODEL --> S2_TRAIN
    VAE_MODEL --> INF
    DIT_MODEL --> S2_TRAIN
    DIT_MODEL --> INF
    ROUTER --> S2_TRAIN
    ROUTER --> INF
    EVAL --> INF

    style VAE_MODEL fill:#e1f5fe
    style DIT_MODEL fill:#fff3e0
    style CONFIG fill:#f5f5f5
    style ATTN fill:#f5f5f5
```

## 创新点映射

| 创新 | 文件 | 核心组件 | 数学基础 |
|------|------|---------|---------|
| **A: Spectral Dual-Latent** | `modeling_mmldm_vae.py` | `fft_decompose` + 双 Conv1d 编码器 | FFT 频域分解 → trend + residual 子空间 |
| **C: TCLR** | `modeling_mmldm_vae.py` | `_compute_tclr()` | 时序对比学习: d_pos < d_neg + margin |
| **TGFM** | `modeling_mmldm_dit.py` | `TextModulator` | 文本 → (scale, shift) 双通路独立调制 |
| **MVTC** | `modeling_mmldm_dit.py` | `MultiViewTextPooler` | K 正交投影 → 文本语义多视图 cycling |
| **FreqLoss** | `training_stage2.py` | `frequency_weighted_flow_loss` | L = L_time + γ_freq·L_FFT + γ_w·Σ(1/k)·|Δz| |
| **BatchMul** | `training_stage2.py` | `repeat_interleave(batch_mul)` | 同样本多 t → 密集训练信号 |
| **CFG** | `training_stage2.py` + `inference.py` | dropout 0.3, guidance_scale 7.0 | v = v_uncond + w·(v_cond - v_uncond) |
| **DCD** | `training_stage2.py` + `modeling_mmldm_dit.py` | `compute_dcd_losses` | 混合潜变量 → 双条件去噪 |
| **LatentStd** | `modeling_mmldm_vae.py` | `standardize_latent` | z_norm = (z - μ_dataset) / σ_dataset |
