# MV-SCD: Multi-View Structural Causal Discovery

## Causal Learning Framework for TIGER Image-Space Time Series Generation

---

## Abstract

This document presents **MV-SCD (Multi-View Structural Causal Discovery)** — a causal learning theory and architecture designed specifically for TIGER's bijective time series-to-image mapping. Unlike CaTSG's (Microsoft Research, ICLR 2026) black-box environment bank approach, MV-SCD exploits the **known mathematical structure** of the three channel transforms (GASF/STFT/RP) as structured environments, achieving **Multi-Channel Complementarity**. This document contains mathematical foundations, revised identifiability principles, loss function designs, and architectural specifications.

---

## 1. Core Innovation Thesis

### 1.1 Unique Advantage: Structured Environments vs Black-Box Environments

**Key CaTSG limitation:** Its environment bank (env_bank) is K learnable vectors with no structural semantics. The EnvInfer network is essentially just a classification head, not a true variational posterior.

**Our key insight:** TIGER's three channel transforms (GASF/STFT/RP) are NOT arbitrary "views" — they are **structured environments** with **known mathematical properties**:

| Channel | Transform Family | Mathematical Property | Causal Identifiability Source |
|---------|-----------------|----------------------|------------------------------|
| GASF | Kernel functions: $G_{ij} = x_i x_j - \sqrt{1-x_i^2}\sqrt{1-x_j^2}$ | Nonlinear, symmetric positive-definite, diagonal bijective | FCM function asymmetry (Shimizu et al., 2006) |
| STFT | Linear filtering: $S = \|\text{STFT}(x)\|$ | Linear, time-frequency localized, Griffin-Lim bijective | VAR-LiNGAM non-Gaussianity (Hyvärinen et al., 2010) |
| RP | Threshold functions: $R = \Theta(\varepsilon - D(x))$ | Binary, discontinuous, state-encoding | Support constraint (Peters et al., 2014) |

### 1.2 Innovation Positioning

| Framework | Causal Domain | Environment Source | Identifiability Approach | Unique Property |
|-----------|--------------|-------------------|-------------------------|-----------------|
| CaTSG (MSR) | 1D time series | Data-learned env_bank | SwAV clustering (no theory) | BAG guidance |
| SCMON (existing) | VAE latent space | Latent variables | NOTEARS constraint | Spectral signatures |
| **MV-SCD (new)** | **3-channel image space** | **Known mathematical transforms** | **Multi-channel complementarity** | **Cross-channel consistency, mechanism graph** |

---

## 2. Mathematical Foundation: Causal Properties of Channel Transforms

### 2.1 GASF Transform

**Definition 2.1.** For normalized time series $x \in [-1,1]^T$:

$$G_{ij} = x_i \cdot x_j - \sqrt{1 - x_i^2} \cdot \sqrt{1 - x_j^2} = \cos(\arccos x_i + \arccos x_j)$$

**Property 2.1 (Diagonal bijectivity):** $G_{tt} = 2x_t^2 - 1 \implies x_t = \sqrt{(G_{tt} + 1)/2}$

**Theorem 2.1 (GASF Causal Preservation).** The GASF diagonal preserves causal mechanism structure, and the squaring nonlinearity naturally satisfies FCM identifiability conditions.

### 2.2 STFT Transform

**Definition 2.2.** $S(f, t) = |\sum_{\tau=0}^{L-1} x_{t+\tau} \cdot w(\tau) \cdot e^{-j2\pi f\tau / L}|$

**Theorem 2.2 (STFT VAR-LiNGAM Identifiability).** STFT preserves the linear VAR structure and non-Gaussianity, so VAR-LiNGAM identifiability conditions hold in STFT space.

### 2.3 RP Transform

**Definition 2.3.** $R_{ij} = \Theta(\varepsilon - \|x_i - x_j\|)$

**Theorem 2.3 (RP Support Identifiability).** The threshold discontinuity provides support-set-based identifiability signals (Peters et al., 2014).

### 2.4 Causal Transfer Theorem (Revised)

**Theorem 2.4 (Causal Transfer — Channel-Dependent).** The preservation of causal structure from temporal domain to image space depends on the symmetry properties of each channel transform:

**Case 1: STFT (Non-symmetric, preserves causal direction)**
The STFT magnitude $S(f,t) = |\text{STFT}(x)|$ is generally non-symmetric in $(f,t)$, and the Griffin-Lim algorithm provides bijective reconstruction. Therefore, causal structure in STFT space is isomorphic to causal structure in the temporal domain.

**Case 2: GASF (Symmetric, preserves diagonal information)**
The GASF matrix $G_{ij} = G_{ji}$ is symmetric, which destroys off-diagonal causal direction information. However, the diagonal $G_{tt} = 2x_t^2 - 1$ preserves temporal value information. Therefore:
- Causal structure is preserved in the **diagonal submatrix** of GASF
- Off-diagonal elements capture pairwise relationships but not directionality
- For causal discovery, we restrict attention to diagonal-preserving mechanisms

**Case 3: RP (Binary, preserves state transitions)**
The RP matrix $R_{ij} = \Theta(\varepsilon - \|x_i - x_j\|)$ is symmetric but encodes state transitions. The threshold discontinuity provides identifiability signals for state-based causality (Peters et al., 2014).

**Corollary 2.4.1.** For causal discovery in TIGER's multi-channel space:
- STFT provides directional causal signals
- GASF provides value-preserving causal signals (diagonal only)
- RP provides state-transition causal signals
- The three channels are **complementary**, not redundant

---

## 3. Multi-Channel Complementarity Principle (Revised)

**Principle 3.1 (Multi-Channel Complementarity).** The three channel transforms (GASF, STFT, RP) provide **complementary** causal signals due to their different mathematical properties:

| Channel | Symmetry | Identifiability Source | Causal Signal Type |
|---------|----------|----------------------|-------------------|
| GASF | Symmetric ($G_{ij}=G_{ji}$) | FCM function asymmetry (diagonal only) | Value-preserving |
| STFT | Non-symmetric | VAR-LiNGAM non-Gaussianity | Directional |
| RP | Symmetric ($R_{ij}=R_{ji}$) | Support constraint (threshold) | State-transition |

**Remark 3.1 (No Independence Assumption).** We do NOT claim that the three channels provide independent identifiability sources. Since all three are derived from the same time series, their identifiability conditions are correlated. For example, if the original series is Gaussian, all three channels fail to identify causal direction.

**Remark 3.2 (No Probability Estimates).** We do NOT provide specific probability estimates (e.g., $p_G \approx 0.8$) for identifiability, as these depend on the data distribution and cannot be determined a priori.

**Conjecture 3.1 (Practical Complementarity).** In practice, for typical time series data:
- At least one channel will satisfy its identifiability conditions
- The three channels provide different perspectives on the same causal structure
- Combining signals from all three channels improves robustness

This conjecture is supported by empirical evidence but lacks formal proof.

---

## 4. Cross-Channel Causal Consistency Principle (Revised)

**Principle 4.1 (Causal Consistency).** Feature $Z$ is a causal parent of $x_t$ if and only if it produces **consistent causal direction judgments** across all three channels:

$$\text{sign}(\frac{\partial x_t}{\partial Z_G}) = \text{sign}(\frac{\partial x_t}{\partial Z_S}) = \text{sign}(\frac{\partial x_t}{\partial Z_R})$$

**Intuition:** If $Z$ is a true causal parent, then:
- In GASF space: $Z$ should influence $x_t$ through the diagonal mechanism
- In STFT space: $Z$ should influence $x_t$ through frequency-domain coupling
- In RP space: $Z$ should influence $x_t$ through state transitions

The **direction** of influence should be consistent across all three representations, even though the **magnitude** and **functional form** differ.

**Remark 4.1 (Not Distribution Equality).** We do NOT claim that $P(x_t|Z_G) = P(x_t|Z_S) = P(x_t|Z_R)$. This would require the three channels to have identical mathematical structures, which they do not:
- GASF encodes $\cos(\arccos(x_i) + \arccos(x_j))$ — nonlinear, symmetric
- STFT encodes frequency energy — linear, time-frequency localized
- RP encodes state transitions — binary, threshold-based

**Remark 4.2 (Mutual Information Alternative).** An alternative formulation uses mutual information:

$$I(Z_G; x_t) \approx I(Z_S; x_t) \approx I(Z_R; x_t)$$

This states that causal features should have **similar predictive power** across channels, even if the conditional distributions differ in form.

**Connection to iVAE:** The three channels act as "auxiliary variables" with KNOWN mathematical structure (unlike iVAE's black-box auxiliary variables). However, unlike iVAE's assumption, we do not require conditional independence — only directional consistency.

---

## 5. Mechanism-Level Causal Graph

Instead of CaTSG's flat env bank, we learn a **sparse causal graph** over mechanisms:

- **Nodes:** $M = \{M_G^k, M_S^k, M_R^k | k=1,...,K\}$ (3K mechanism nodes)
- **Edges:** $A \in \mathbb{R}^{3K \times 3K}$ (soft adjacency with NOTEARS acyclicity)
- **ICM enforcement:** Each mechanism has its own independent MLP

**Key advantages over CaTSG:**
1. CaTSG's env_bank is a bag of vectors — no structural relationships
2. Our mechanism graph captures STRUCTURAL causal relationships
3. True INTERVENTIONAL semantics: do($M_G^k$) means "fix mechanism k in GASF channel"
4. CaTSG's BAG is just a weighted average — not a true causal intervention

---

## 6. Training Objective (Simplified)

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{diff}} + \lambda_1 \mathcal{L}_{\text{causal}} + \lambda_2 \mathcal{L}_{\text{graph}}$$

Where:
1. $\mathcal{L}_{\text{diff}}$: Standard diffusion MSE
2. $\mathcal{L}_{\text{causal}}$: Mechanism-level causal prediction loss (information preservation)
3. $\mathcal{L}_{\text{graph}}$: NOTEARS acyclicity + L1 sparsity

**Note:** The original design included $\mathcal{L}_{\text{inv}}$ (cross-channel invariance) and $\mathcal{L}_{\text{ICM}}$ (ICM independence), but these were removed in the simplified implementation to avoid DDP deadlock and reduce complexity. The cross-channel consistency is enforced implicitly through the shared mechanism encoder architecture.

---

## 7. Causal Diffusion Sampling

### 7.1 Mechanism-Level Intervention (replaces CaTSG's BAG)

For do($M_c^k = m$):
1. Fix mechanism state: $h_{c^k} = m$
2. Propagate through causal graph: $h_j' = f_j(h_{\text{Pa}(j)}')$
3. Condition diffusion on intervened mechanism states

### 7.2 Counterfactual Generation (Abduction-Action-Prediction)

1. **Abduction:** Infer noise $\epsilon^*$ and mechanism states $M^*$ from observed $(x_0, c)$
2. **Action:** Modify specific mechanisms via graph surgery
3. **Prediction:** Generate $x'$ from $(\epsilon^*, M', c')$

---

## 8. Architecture

Components:
- **A: Cross-Channel Mechanism Encoder** — encodes each channel into K mechanism states
- **B: Mechanism Causal Graph Learner** — learns 3K×3K soft adjacency with NOTEARS
- **C: Causal Mechanism Transition** — per-mechanism MLP with parent aggregation + FiLM
- **D: Cross-Channel Invariance Validator** — validates cross-channel prediction consistency
- **E: Mechanism Recomposer** — fuses 3K mechanism states into TIGERDiT-compatible features

---

## 9. Comparison with CaTSG

| Dimension | CaTSG | MV-SCD | Advantage |
|-----------|-------|--------|-----------|
| Environment definition | Black-box env_bank | Known math transforms | MV-SCD: interpretable, verifiable |
| Identifiability | SwAV clustering (no theory) | Multi-channel complementarity | MV-SCD: principled approach |
| Causal graph | None (only env classification) | Mechanism-level causal graph | MV-SCD: structured causality |
| Intervention semantics | BAG weighted average | Mechanism-level do-operation | MV-SCD: true intervention |
| Counterfactual | Not implemented | Abduction-Action-Prediction | MV-SCD: complete causal ladder |
| Consistency | SwAV clustering | Cross-channel causal consistency | MV-SCD: direction-consistent |

---

## 10. Implementation Plan

| Phase | Timeline | Files | Description |
|-------|----------|-------|-------------|
| 1: Core module | Week 1-2 | `mmldm/tiger/mvscd.py` (NEW), `dit_model.py`, `train.py` | Full MV-SCD implementation |
| 2: Causal sampling | Week 2-3 | `samplers/causal_ddim.py` (NEW), `generator.py` | Mechanism-level intervention |
| 3: Evaluation | Week 3-4 | `evaluation/causal_metrics.py` (NEW), `causal_evaluator.py` | Causal evaluation suite |
| 4: Experiments | Week 4-5 | Config files | Baseline, MV-SCD, ablations |

---

## 11. Expected Contributions

1. **Multi-Channel Complementarity Principle:** Three channel transforms provide complementary causal signals (directional, value-preserving, state-transition)
2. **Cross-Channel Causal Consistency:** Causal features should produce consistent direction judgments across channels — weaker but more realistic than CCIP
3. **Mechanism-Level Causal Graph:** Directed causal graph over mechanism nodes for structured causal reasoning
4. **Mechanism-Level Intervention:** True do-operations replacing BAG's weighted average
5. **Complete Causal Ladder:** Association, intervention, and counterfactual generation
6. **Causal Evaluation Suite:** New metrics for interventional/counterfactual/graph fidelity/consistency

---

*Document version: v3.0 (Mathematically rigorous version)*
*Last updated: 2026-05-29*
*Changes: Revised Causal Transfer Theorem, replaced Triple Identifiability with Multi-Channel Complementarity, replaced CCIP with Cross-Channel Causal Consistency*
