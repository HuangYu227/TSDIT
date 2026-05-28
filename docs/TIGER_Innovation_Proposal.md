# TIGER 三大创新方向方案

**日期**: 2026-05-28
**分支**: newtime
**作者**: AI Research Assistant

---

## 一、多模态融合创新：Channel-Structure-Aware Fusion（通道结构感知融合）

### 核心问题

当前MMDiT对三个通道（GASF/STFT/RP）做**统一处理**，但它们的数学结构完全不同：
- **GASF**：T×T Gramian矩阵，半正定，捕获时间点对相关性
- **STFT**：频率×时间谱图，Hermitian对称，捕获频域能量分布
- **RP**：二值递归矩阵，稀疏，捕获动力学状态转移

### 推荐方案：CSA-MoE + SCCA（通道结构感知混合专家 + 结构一致性交叉注意力）

#### 创新点1：通道结构感知混合专家（CSA-MoE）

受 Diff-MoE (ICML 2025)、MoS (arXiv 2511.12207) 启发，不同通道应由**不同架构的专家**处理：

```python
class ChannelStructureRouter(nn.Module):
    """基于通道类型和时间步的路由"""
    def __init__(self, dim, num_experts=4):
        super().__init__()
        self.channel_embed = nn.Embedding(4, dim)  # 3通道 + 跨通道
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim), nn.SiLU(),
            nn.Linear(dim, num_experts)
        )

    def forward(self, x, channel_ids, t_emb):
        c_emb = self.channel_embed(channel_ids)
        gate_input = torch.cat([x, c_emb], dim=-1) + t_emb.unsqueeze(1)
        return self.gate(gate_input)
```

关键：专家不是同质的MLP，而是**结构特化的**：
- GASF专家：行-列分解注意力（捕获T×T矩阵结构）
- STFT专家：频带注意力（沿频率轴聚合）
- RP专家：稀疏递归注意力（利用二值稀疏性）
- 跨通道专家：对齐三个通道的结构对应关系

#### 创新点2：结构一致性交叉注意力（SCCA）

三个通道编码**同一个时间序列**，存在确定性的数学对应关系：
- GASF对角线 ↔ 时间序列值
- STFT频率峰值 ↔ 周期性成分
- RP对角带 ↔ 状态持续性

```python
class StructuralConsistencyModule(nn.Module):
    """强制三个通道在去噪过程中保持结构一致性"""
    def __init__(self, dim):
        super().__init__()
        self.cross_channel_attn = nn.ModuleDict({
            'gasf_stft': nn.MultiheadAttention(dim, num_heads=4, batch_first=True),
            'stft_rp': nn.MultiheadAttention(dim, num_heads=4, batch_first=True),
            'rp_gasf': nn.MultiheadAttention(dim, num_heads=4, batch_first=True),
        })

    def forward(self, channel_features):
        gasf_refined, _ = self.cross_channel_attn['gasf_stft'](*)
        stft_refined, _ = self.cross_channel_attn['stft_rp'](*)
        rp_refined, _ = self.cross_channel_attn['rp_gasf'](*)
        return [gasf_refined, stft_refined, rp_refined]
```

**发表角度**：第一个将多通道时间序列图像的通道视为**结构异构模态**而非简单RGB通道的工作。适合 ICLR/NeurIPS。

**参考文献**:
- Diff-MoE (Cheng et al., ICML 2025)
- MoS - Mixture of States (arXiv 2511.12207, 2025)
- FuseMoE (NeurIPS 2024)
- TACA (Lv et al., ICCV 2025)
- FCDiffusion (CVPR 2025)
- FreSca (arXiv 2504.02154)

---

## 二、因果学习创新：从关联到反事实的因果生成

### 核心问题

当前SCMON停留在Pearl因果层级的**L1（关联层）**：
- 频谱兼容性发现因果图 → 仅是关联发现
- NOTEARS约束 `tr(exp(W∘W)) - K = 0` → 仅保证无环性
- 无法回答 `P(Y|do(X))` 或反事实查询

### 数学基础：Pearl因果层级（PCH）

Pearl (2009) 定义三层：
- **L1（关联）**：P(Y|X) — 观测条件概率
- **L2（干预）**：P(Y|do(X=x)) — 干预分布
- **L3（反事实）**：P(Y_{X=x'} | X=x, Y=y) — 单元级假设

**关键定理**（Bareinboim et al., 2022）：低层数据**原则上不能**识别高层因果量，除非增加假设。

### 推荐方案：Causal SCMON (C-SCMON)

基于 DCM (Chao et al., TMLR 2024)、CaTSG (Xia et al., 2025)、Diff-SCM (Sanchez et al., 2022) 的数学框架。

#### Stage 1：因果图发现（增强SCMON）

保留现有的频谱分解 + NOTEARS，增加：
- **ICP-TS验证**（Peters et al., 2016）：测试 P(z_k | PA_k) 在不同时间片段上的不变性
- **稀疏性约束**（IDOL, 2024）：保证机制子空间的可辨识性

#### Stage 2：神经结构方程（扩散机制）

每个机制 k 的结构方程建模为**条件扩散模型**：

z_k^t = g_θ^k(PA_k(z^{t-1}), u_k^t)

训练损失：

L_k = E[||ε - ε_θ^k(z_k^t, PA_k, t)||^2] + λ_1 · KL(q(u_k) || N(0,I)) + λ_2 · ||z_k^t - g_θ^k(PA_k, u_k^t)||^2

#### Stage 3：因果查询应答

**观测生成（L1）**：标准反向扩散

**干预生成（L2）**：截断因子分解（Pearl, 2009, Theorem 3.2.1）

P(z | do(z_k = c)) = ∏_{j≠k} p(z_j | PA_j) · δ(z_k - c)

**反事实生成（L3）**：Abduction-Action-Prediction
1. Abduction：用DDIM编码恢复外生噪声 u_k = DDIM_encode(z_k^F | PA_k^F)
2. Action：干预 z_k = c
3. Prediction：用固定U按拓扑序重新生成

#### 数学性质

- **干预正确性**：因果Markov条件下，截断因子分解给出正确干预分布（Pearl, Theorem 1.4.1）
- **反事实一致性**：扩散机制收敛到真实结构方程时，反事实估计收敛到SCM反事实（Chao et al., 2023）
- **可辨识性**：稀疏性+时序结构+充分变异性下，机制子空间可辨识到Markov等价类（IDOL, 2024）

#### 评估指标

- **CED**：||E[Y|do(X=x)]_gen - E[Y|do(X=x)]_true||_2
- **CV**：反事实是否满足所有结构方程
- **AE**：||D(E(x^F)) - x^F||_2（应趋近0）
- **SHD**：图恢复的Structural Hamming Distance

#### 各方法与SCMON对比

| 方法 | Pearl层级 | 关键数学 | 与SCMON差异 |
|------|-----------|---------|------------|
| DCM (TMLR'24) | L1+L2+L3 | 每节点扩散SCM | SCMON无每节点扩散；无abduction |
| CaTSG (2025) | L1+L2+L3 | Backdoor调整得分引导 | SCMON无干预引导 |
| CausalDiffAE (2024) | L1+L3 | 因果编码+DDIM反事实 | SCMON无因果编码损失 |
| Diff-SCM (2022) | L1+L2+L3 | 基于得分的SCM | SCMON算子不是得分网络 |
| DoFlow (ICLR'25) | L1+L2+L3 | 可逆CNF over DAG | SCMON使用非可逆算子 |
| DiffPO (NeurIPS'24) | L2 | 正交扩散损失 | SCMON无倾向性校正 |
| ICP-TS (2016) | L1 | 不变性测试 | SCMON用频谱兼容性而非不变性 |
| BELM-MDCM (2025) | L3 | CIC/零SRE | SCMON的DDIM编码有非零SRE |

**发表角度**：第一个在扩散框架中实现Pearl三层因果层级全部应答的时间序列生成模型。适合 NeurIPS/ICML。

**关键参考文献**:
- Pearl (2009), *Causality*, Cambridge
- Bareinboim et al. (2022), "Pearl's Causal Hierarchy", Foundations and Trends in AI
- Chao et al. (2024), TMLR — DCM
- Xia et al. (2025) — CaTSG, Microsoft Research
- Komanduri et al. (2024) — CausalDiffAE
- Sanchez et al. (2022) — Diff-SCM
- DoFlow (ICLR 2025)
- DiffPO (NeurIPS 2024)
- BELM-MDCM (2025)
- Peters et al. (2016) — ICP-TS
- IDOL (2024)

---

## 三、LLM集成创新：LLM作为时间序列生成的推理引擎

### 核心问题

当前CLIP/文本编码器将文本压缩为**静态向量**，丢失了：
- 时间顺序信息（"先上升后下降"）
- 因果推理（"因为温度升高所以湿度增加"）
- 多尺度结构（趋势 vs 周期 vs 噪声）

### 推荐方案：LLM-TIGER（三角色LLM集成）

#### 角色1：时间分解器（Pre-generation Planning）

LLM将用户文本分解为结构化时间蓝图，**不同扩散时间步接收不同粒度的分解信息**。

#### 角色2：迭代批评者（Post-generation Critique）

生成后，LLM检查时间序列是否符合原始描述，产生修正信号指导二次生成。

#### 角色3：因果/体制规划器（Causal Regime Planning）

LLM识别文本中的因果结构和体制转换，生成离散规划作为结构化条件。

#### 与竞争对手对比

| 系统 | 文本编码器 | LLM角色 | 迭代？ | 结构化规划？ |
|------|-----------|---------|--------|------------|
| T2S (IJCAI'25) | CLIP | 无 | 否 | 否 |
| Diffusion-TS (ICLR'24) | 无 | 无 | 否 | 否 |
| VerbalTS | CLIP | 无 | 否 | 否 |
| **TIGER-LLM (拟)** | **CLIP + LLM** | **分解+批评+规划** | **是** | **是** |

**发表角度**：第一个将LLM推理（而非仅编码）用于时间序列生成的框架。适合 AAAI/IJCAI。

**参考文献**:
- Time-LLM (Jin et al., ICLR 2024)
- TimeOmni-1 (Guan et al., 2025)
- Time-MQA (Kong et al., ACL 2025)
- DALL-E 3 (Betker et al., 2023)
- Chronos (Ansari et al., 2024)
- TimesFM (Google, ICML 2024)

---

## 四、三个创新点的协同关系

```
用户文本 → [LLM分解器] → 结构化时间蓝图
                              ↓
                    [因果SCMON] → 因果图 + 机制分解
                              ↓
                    [CSA-MoE融合] → 通道特化处理
                              ↓
                    [扩散生成] → 时间序列图像
                              ↓
                    [LLM批评] → 修正 → 重新生成
                              ↓
                    [图像→时间序列] → 最终输出
```

## 五、推荐实施顺序

1. **因果学习（C-SCMON）** — 最强创新性 + 严格数学基础，可独立发表
2. **多模态融合（CSA-MoE + SCCA）** — 工程创新，可与因果学习结合
3. **LLM集成（LLM-TIGER）** — 应用创新，依赖前两者的架构

每个方向都可以独立成文，也可以组合成一个完整系统。
