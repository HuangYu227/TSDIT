# CIED: 跨通道因果干预环境发现

## 面向 TIGER 图像空间时间序列生成

---

## 1. 核心创新论点

### 1.1 独特优势：双射时间序列-图像因果传递

本框架的双射 TS <-> Image 映射赋予了其他 TSG 框架所不具备的性质：

> **因果传递定理**：在三通道图像空间中发现的因果结构与时间域中的因果结构同构。

**证明概要：**
- GASF 对角线直接编码原始时间序列值：`x_t = sqrt((G_tt + 1) / 2)`
- STFT 列编码时间频率演化（Griffin-Lim 双射）
- RP 编码动力学状态转换（基于距离）
- 因此：因果关系 `STFT_channel[f,t] -> GASF_channel[t]` 意味着"频率分量 f 在时刻 t 导致值变化"——这是一个真实的时间域因果声明。

### 1.2 创新性分析

| 框架 | 因果域 | 因果机制 | 独特性质 |
|------|--------|---------|---------|
| CaTSG (MSR) | 1D 时间序列 | 环境库后门调整 | 干预/反事实 TSG |
| SCMON (已有) | VAE 潜空间 | 频谱机制分解 | 频谱因果签名 |
| **CIED (新)** | **3通道图像空间** | **跨通道因果图 + 环境发现** | **双射因果传递、多视角因果** |

**与 CaTSG 的关键差异：**
1. **多视角因果**：CaTSG 将时间序列视为 1D。CIED 利用 GASF/STFT/RP 提供同一时间过程的 3 个互补"视角"，实现跨视角因果发现。
2. **图像空间效率**：对 2D patch token 的因果操作比 1D 序列更具可并行性。
3. **结构归纳偏置**：通道结构（GASF=对称、STFT=局部、RP=二值）为因果机制设计提供了领域先验。

---

## 2. 理论基础

### 2.1 三通道图像的结构因果模型

在图像表示上定义 SCM：

```
变量：
  G = GASF 通道 (R)        -- 编码值相关性
  S = STFT 通道 (G)        -- 编码频谱演化
  R = 递归图通道 (B)        -- 编码状态转换
  E = 潜在环境变量          -- 未观测混杂因子
  C = 文本条件              -- 上下文

因果图（可学习）：
  E -> G, E -> S, E -> R    （环境影响所有通道）
  C -> G, C -> S, C -> R    （文本条件影响所有通道）
  G <-> S                    （值-频率双向因果）
  S <-> R                    （频率-状态双向因果）
  G <-> R                    （值-状态双向因果）

干预分布：
  P(G, S, R | do(C=c)) = Σ_e P(G|S,R,c,e) · P(S|G,R,c,e) · P(R|G,S,c,e) · P(e)

反事实：
  给定观测 (G_0, S_0, R_0, c)，推断 e* = argmax P(e|G_0,S_0,R_0,c)
  然后在 do(C=c') 下以固定 e* 生成 (G', S', R')
```

### 2.2 与 Pearl 因果阶梯的对应关系

| 层级 | 任务 | CIED 实现 |
|------|------|----------|
| L1 (关联) | P(X\|C) | 标准 TIGER 扩散 |
| L2 (干预) | P(X\|do(C)) | 跨通道环境库 BAG 引导扩散 |
| L3 (反事实) | P(X'\|do(C'), 观测 X,C) | 固定环境后验的 推断-行动-预测 |

---

## 3. 架构设计

### 3.1 模块总览

```
                    +-----------------------------------------+
                    |           CIED 模块                      |
                    |                                         |
  图像 patches -->  |  +---------------------------------+   |
  (B, 3, H, W)     |  |  跨通道 Patch 编码器              |   |
                    |  |  （每通道独立编码器）              |   |
                    |  +----------+----------------------+   |
                    |             |                            |
                    |             v                            |
                    |  +---------------------------------+   |
                    |  |  因果图学习器                     |   |
                    |  |  （3K x 3K 软邻接矩阵）           |   |
                    |  |  NOTEARS 无环性约束               |   |
                    |  +----------+----------------------+   |
                    |             |                            |
                    |             v                            |
                    |  +---------------------------------+   |
  文本 emb -------> |  |  环境推断网络                     |   |
                    |  |  （多视角 EnvInfer）              |   |
                    |  |  -> env_probs (K,)               |   |
                    |  +----------+----------------------+   |
                    |             |                            |
                    |             v                            |
                    |  +---------------------------------+   |
                    |  |  因果机制转换                     |   |
                    |  |  （每机制 MLP + 父节点聚合）       |   |
                    |  +----------+----------------------+   |
                    |             |                            |
                    |             v                            |
                    |  +---------------------------------+   |
                    |  |  跨通道重组器                     |   |
                    |  |  （通道感知交叉注意力）            |   |
                    |  +----------+----------------------+   |
                    |             |                            |
                    |             v                            |
  causal_features --|---> (B, channels, 1, total_tokens)      |
  causal_losses ----|---> 辅助损失字典                         |
                    +-----------------------------------------+
```

### 3.2 组件 A：跨通道 Patch 编码器

**目的**：独立编码每个通道以捕获通道特有结构，然后形成机制组。

```python
class CrossChannelPatchEncoder(nn.Module):
    """将每个图像通道编码为机制级表示。

    对每个通道 c ∈ {GASF, STFT, RP}：
        1. 提取通道：x_c = image[:, c:c+1, :, :]
        2. Patch 嵌入：(B, 1, H, W) -> (B, d_model, n_h, n_w)
        3. 机制分组：将 patches 软分配到 K_c 个机制子空间
        4. 每机制池化：(B, d_model, K_c)

    输出：mechanism_states，形状 (B, 3*K_total, d_model)
    """
```

**设计选择：**
- 每个通道拥有独立的 patch 编码器（不同归纳偏置：GASF 对称、STFT 时频、RP 二值）
- 机制分组使用可学习原型（类似 SCMON 的子空间分解器）
- 默认：每通道 K=4 个机制，共 12 个

### 3.3 组件 B：因果图学习器

**目的**：在 3K 个机制节点上学习有向因果图。

```python
class CrossChannelCausalGraphLearner(nn.Module):
    """学习 3K x 3K 软因果邻接矩阵。

    关键创新：通道感知初始化。
    - 通道内边初始化为 0（通道内因果较弱）
    - 通道间边初始化为小随机值
    - 这将发现偏向跨通道因果

    损失：
        - NOTEARS：tr(exp(W ⊙ W)) - 3K = 0  （无环性）
        - L1 稀疏性：λ ||W||_1
        - 通道一致性：若 G->S 边存在，S->G 应具有
          兼容强度（双向关系的软对称性）
    """
```

**关键创新——通道感知因果先验：**
- 3 个通道具有领域知识中的已知结构关系：
  - GASF 编码值 -> STFT 编码频率：值变化导致频率变化（G->S 先验）
  - STFT 编码频率 -> RP 编码状态：频率偏移导致状态转换（S->R 先验）
  - RP 编码状态 -> GASF 编码值：状态转换导致值变化（R->G 先验）
- 这些被编码为邻接矩阵上的**可学习先验偏置**，而非硬约束

### 3.4 组件 C：环境推断网络 (EnvInfer)

**目的**：从多视角图像 patch 推断潜在环境/混杂因子。

```python
class CrossChannelEnvInfer(nn.Module):
    """从三通道图像 patch 进行多视角环境推断。

    架构：
        1. 每通道特征提取（空洞卷积 + 注意力池化）
        2. 跨通道特征融合（拼接 + MLP）
        3. 统计特征：每通道的均值、标准差、最大值
        4. 频谱特征：每通道表示的 FFT
        5. 与 env_bank 评分：点积 / τ -> softmax

    与 CaTSG EnvInfer 的关键差异：
        - CaTSG：输入为 [x; c]（1D 时间序列 + 上下文）
        - CIED：输入为 [G; S; R]（3通道图像 patch）
        - CIED 利用多视角结构实现更好的环境判别

    输出：env_probs，形状 (B, K)，K 为环境数量
    """
```

### 3.5 组件 D：因果机制转换

**目的**：对每个机制应用因果干预，遵循学习到的图结构。

```python
class CausalMechanismTransition(nn.Module):
    """带因果父节点聚合的每机制转换。

    对每个机制 m：
        1. 从因果图获取父节点：Pa(m) = {j : W[j,m] > threshold}
        2. 聚合父节点状态：h_parent = Σ_j W[j,m] · h_j
        3. 转换：h_m' = MLP_m(concat(h_m, h_parent))
        4. 环境调节：h_m'' = h_m' · (1 + γ_e) + β_e
           其中 γ_e, β_e 为环境条件 FiLM 参数

    ICM 原理：每个机制拥有独立 MLP（可独立修改）
    """
```

### 3.6 组件 E：跨通道重组器

**目的**：将机制状态融合回 TIGERDiT 的单一调节信号。

```python
class CrossChannelRecomposer(nn.Module):
    """将 3K 个机制状态融合为 (B, channels, 1, total_tokens)。

    设计：
        1. 通道内交叉注意力：融合每通道内的 K 个机制
        2. 跨通道交叉注意力：融合 3 个通道摘要
        3. 可学习重要性门控（零初始化以确保安全启动）
        4. 投影到 TIGERDiT token 空间

    输出：与 TIGERDiT ResidualBlock 输入兼容的 causal_features
    """
```

---

## 4. 集成到 TIGER 框架

### 4.1 注入点：TIGERDiT.forward() 内部

```python
# 在 TIGERDiT.forward() 中，多 patch 编码（步骤 3）之后，ResidualBlock（步骤 6）之前：

# --- 新增：CIED 因果特征注入 ---
if self.cied is not None:
    cied_output = self.cied(image, attr_emb)  # 原始图像 + 文本条件
    causal_features = cied_output.causal_features  # (B, channels, 1, total_tokens)
    causal_losses = cied_output.losses

    # 门控注入（零初始化以确保安全训练启动）
    x_in = x_in + self.causal_gate * causal_features
```

### 4.2 注入点：扩散采样（BAG 引导）

对于干预和反事实生成，修改采样循环：

```python
# 在 TIGERGenerator.generate() 中：

# 标准扩散：eps = dit(x_t, t, attr_emb)
# CIED 引导扩散：
#   1. 对 (x_t, attr_emb) 运行 EnvInfer -> env_probs w_k
#   2. 用每个 env 条件运行 dit K 次 -> eps_k
#   3. BAG：eps = (1+ω) · Σ(w_k · eps_k) - ω · eps_base
```

### 4.3 训练损失

```
L_total = L_diffusion + λ_causal · L_causal + λ_env · L_env

其中：
  L_diffusion = MSE(noise, noise_pred)           # 标准扩散损失
  L_causal = L_notars + L_sparsity + L_spectral  # CIED 因果损失
  L_env = L_swav + L_orthogonality               # 环境发现损失
```

---

## 5. 因果评估指标（新增）

### 5.1 干预分布匹配

```python
def compute_interventional_metrics(real_ts, gen_ts, context, do_context):
    """比较 P(X|do(C)) vs P_true(X|do(C))。

    对每个 do 上下文 c'：
        1. 以 c' 为条件生成 N 个样本
        2. 与相同干预下的真实样本分布比较
        3. 指标：MMD-RBF、KL 散度、J-FTSD
    """
```

### 5.2 反事实一致性

```python
def compute_counterfactual_metrics(observed_ts, cf_gen_ts, cf_context):
    """衡量反事实质量。

    给定观测 (X, C) 和反事实上下文 C'：
        1. 生成 X' = CF(X, C, C')
        2. 检查：X' 应仅在 C' 与 C 不同之处与 X 不同
        3. 指标：CF-MSE、CF-H@1、机制一致性分数
    """
```

### 5.3 因果图保真度

```python
def compute_graph_metrics(learned_graph, ground_truth_graph=None):
    """评估发现的因果结构。

    若有 ground truth：
        - 结构汉明距离 (SHD)
        - 边检测 F1
        - 边权重 AUROC

    若无 ground truth（真实数据）：
        - 图稀疏性（应稀疏）
        - 无环性违反（应 ≈ 0）
        - bootstrap 重采样下的稳定性
    """
```

### 5.4 机制模块化分数

```python
def compute_mechanism_modularity(mechanism_states, causal_graph):
    """评估 ICM 原理遵循度。

    对每个机制 m：
        1. 扰动机制 m 的输入
        2. 衡量其他机制的变化
        3. 好的模块化：扰动仅影响因果图中的子节点
    """
```

---

## 6. 实施计划

### 阶段 1：核心 CIED 模块（第 1-2 周）

| 文件 | 操作 | 描述 |
|------|------|------|
| `mmldm/tiger/cied.py` | **新建** | 完整 CIED 模块实现 |
| `mmldm/tiger/dit_model.py` | 修改 | 在 TIGERDiT 中添加 CIED 注入 |
| `mmldm/tiger/generator.py` | 修改 | 添加 BAG 引导采样 |
| `mmldm/tiger/train.py` | 修改 | 在训练循环中添加因果损失 |

### 阶段 2：因果采样（第 2-3 周）

| 文件 | 操作 | 描述 |
|------|------|------|
| `mmldm/tiger/samplers/causal_ddim.py` | **新建** | 带 BAG 的因果 DDIM 采样器 |
| `mmldm/tiger/samplers/causal_ddpm.py` | **新建** | 带 BAG 的因果 DDPM 采样器 |

### 阶段 3：评估（第 3-4 周）

| 文件 | 操作 | 描述 |
|------|------|------|
| `mmldm/tiger/evaluation/causal_metrics.py` | **新建** | 所有因果评估指标 |
| `mmldm/tiger/evaluation/causal_evaluator.py` | **新建** | 端到端因果评估 |

### 阶段 4：实验（第 4-5 周）

| 实验 | 配置 | 目的 |
|------|------|------|
| 基线（无 CIED） | 现有配置 | 标准 TIGER 性能 |
| CIED（仅训练） | 新配置 | 因果特征 + 环境发现 |
| CIED + BAG 采样 | 新配置 | 完整干预生成 |
| CIED + 反事实 | 新配置 | 反事实生成 |
| 消融：无环境库 | 新配置 | 环境发现的效果 |
| 消融：无因果图 | 新配置 | 因果结构的效果 |

---

## 7. 预期贡献

1. **新颖的任务定义**：首次在图像空间中定义具有双射传递保证的因果 TSG
2. **跨通道因果发现**：发现 GASF/STFT/RP 通道间因果关系的新机制
3. **多视角环境推断**：利用三通道结构实现优于 1D 方法的混杂因子检测
4. **因果评估套件**：干预/反事实 TSG 质量的新指标
5. **实证验证**：证明因果结构在标准基准上提升生成质量

---

## 8. 关键参考文献

| 论文 | 相关性 | 借鉴内容 |
|------|--------|---------|
| CaTSG (arXiv:2509.20846) | 主要参考 | 后门调整、EnvInfer、BAG 公式 |
| CausalVAE Plug-in (arXiv:2604.07712) | 架构 | 模块化因果层设计 |
| Mask2Cause (arXiv:2605.07280) | 注意力 | 邻接约束注意力 |
| PTCD (arXiv:2605.26759) | 训练 | 因果增强策略 |
| SCMON (现有代码库) | 基础 | 机制分解、频谱签名 |
| Function-Valued Causal (arXiv:2605.26408) | 评估 | ICE 曲线比较 |
