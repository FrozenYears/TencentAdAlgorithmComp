# 广告算法技术路线 - CTR/CVR预测最新进展

> 整理时间：2026年5月  
> 适用场景：腾讯广告算法大赛2026技术储备

---

## 一、CTR预测模型演进

### 1.1 经典模型

CTR（Click-Through Rate）预测模型经历了从浅层到深层的演进过程：

| 阶段 | 模型 | 核心思想 | 特点 |
|------|------|---------|------|
| 浅层模型 | LR | 逻辑回归 | 简单高效，依赖特征工程 |
| 二阶交叉 | FM | 因子分解机 | 自动学习二阶特征交叉 |
| 深度交叉 | DeepFM | FM + DNN | 兼顾低阶和高阶特征交互 |
| 显式交叉 | DCN | Cross Network | 显式学习有界度交叉特征 |
| 组合交叉 | xDeepFM | CIN + DNN | 压缩交互网络，向量级交叉 |
| 注意力机制 | DIN | 注意力激活 | 根据候选商品动态激活兴趣 |
| 兴趣演化 | DIEN | GRU + 注意力 | 建模用户兴趣演化过程 |
| 自注意力 | AutoInt | Multi-Head Attention | 自动学习特征交互 |
| 二阶DCN | DCN V2 | 混合交叉结构 | 低秩近似提升表达能力 |

### 1.2 最新研究（2024-2026）

近年来CTR预测领域涌现出多项重要进展：

**DCNv3**
- 核心创新：线性+指数级交叉，Self-Mot降噪机制
- 突破点：同时实现显式和隐式特征交叉

**DSAIN**
- 全称：Deep Scenario-Aware Interaction Network
- 核心创新：情境感知交互网络
- 效果：CTR提升2.70%，已开源

**ELEC**
- 全称：Enhancing CTR Prediction with LLM
- 核心创新：LLM增强CTR预测，知识蒸馏
- 意义：打通大模型与推荐系统

**DCIN**
- 全称：Deep Contextual Interest Network
- 核心创新：位置感知上下文聚合
- 应用：上下文信息建模

**FDIIIN**
- 全称：Dual Importance-Aware Interaction Network
- 核心创新：双重重要性感知网络

---

## 二、CVR预测方法

### 2.1 核心挑战

CVR（Conversion Rate）预测面临三个核心挑战：

| 挑战 | 说明 | 影响 |
|------|------|------|
| 样本选择偏差（SSB） | CVR训练样本仅来自点击样本，存在分布偏差 | 模型泛化能力受限 |
| 数据稀疏性（DS） | 转化样本远少于点击样本 | 模型难以学习有效模式 |
| 延迟反馈 | 转化行为可能延迟发生 | 样本标签不准确 |

### 2.2 关键模型

**ESMM**
- 全称：Entire Space Multi-Task Model
- 核心公式：`pCTCVR = pCTR × pCVR`
- 创新：在全空间建模，解决样本选择偏差
- 地位：多任务CVR预测的里程碑模型

**ESCM2**
- 全称：Entire Space Counterfactual Multi-Task Model
- 核心创新：反事实风险最小化
- 突破：从因果推断角度解决SSB问题

**TESLA**
- 全称：Time-Aware Entire Space Learning for Delayed Feedback Modeling
- 核心创新：级联延迟反馈建模
- 地位：NIPS 2024论文，工业级方案

**MAL**
- 全称：Multi-Attribution Learning
- 核心创新：多归因学习
- 效果：GAUC提升0.51%

**TRACE**
- 全称：Trajectory Conditioned Delay Feedback Model
- 核心创新：轨迹条件延迟反馈建模
- 已开源

**DHEN**
- 全称：Deep Hierarchical Ensemble Network
- 核心创新：深度层次集成网络
- 应用：Meta大规模广告系统

---

## 三、多任务学习

### 3.1 经典架构

多任务学习架构的演进：

| 架构 | 特点 | 适用场景 |
|------|------|---------|
| Shared-Bottom | 共享底层网络 | 任务相关性高 |
| MMoE | 混合专家门控 | 任务相关性低 |
| PLE | 渐进式分层萃取 | 多任务优化 |
| ESMM | 全空间多任务 | CVR预测 |
| AITM | 自适应信息迁移 | 任务依赖建模 |

### 3.2 最新进展

**SMES**
- 全称：Scalable Mixture of Experts for Sparse Models
- 核心创新：可扩展稀疏MoE框架
- 实践：快手部署4亿+日活用户系统
- 意义：工业级大规模MoE的典范

**Hetero-MMoE**
- 核心创新：异构专家（MLP + DCN + CIN混合）
- 实践：Uber推荐系统
- 突破：打破同构专家限制

**CAMoE**
- 全称：Cross-Modal Adaptive MoE
- 核心创新：跨模态自适应MoE
- 应用：Spotify音频广告推荐

---

## 四、用户行为序列建模

### 4.1 技术演进

用户行为序列建模是CTR/CVR预测的核心技术：

```
DIN (注意力激活)
  ↓
DIEN (兴趣演化)
  ↓
DSIN (会话兴趣)
  ↓
BST (Transformer序列)
  ↓
SIM (超长序列)
  ↓
TWIN (双层序列)
  ↓
HSTU (生成式排序)
  ↓
HyFormer/GRAB (最新前沿)
```

### 4.2 最新模型

**HyFormer**
- 全称：Unified Hybrid Transformer
- 核心创新：Query Decoding + Query Boosting双机制
- 突破：统一不同序列建模范式

**GRAB**
- 全称：Generative Ranking Framework
- 实践：百度搜索/广告
- 效果：CTR提升3.49%
- 意义：生成式排序的工业实践

**CADET**
- 全称：Contextual Conditioned Decoder
- 实践：LinkedIn推荐系统
- 效果：CTR提升11.04%
- 核心：上下文条件解码器

**UxSID**
- 全称：Semantic-Aware User Interest Modeling
- 应用：用户兴趣语义建模
- 效果：广告收入提升0.337%

**腾讯广告长序列实践**
- 场景：微信广告
- 效果：GMV提升4.22%
- 意义：超长序列建模的工业验证

---

## 五、大语言模型应用

### 5.1 主要方向

LLM在广告推荐领域的三大应用方向：

| 方向 | 代表模型 | 应用场景 |
|------|---------|---------|
| LLM增强CTR预测 | ELEC、CTRL、FLIP | 提升预测精度 |
| 生成式排序 | GRAB、CADET、HSTU | 统一排序范式 |
| 语义理解增强 | 冷启动、内容理解 | 解决稀疏问题 |

### 5.2 工程实践

**知识蒸馏**
- 流程：LLM → 轻量级模型
- 优势：兼顾效果和效率
- 代表：ELEC框架

**特征增强**
- 方式：LLM生成语义特征
- 应用：物品描述、用户画像增强
- 效果：缓解冷启动问题

**预训练-微调**
- 方法：对比学习预训练
- 目标：学习通用语义表示
- 优势：迁移学习能力

---

## 六、开源资源

### 6.1 综合框架

| 框架 | 技术栈 | Stars | 特点 |
|------|--------|-------|------|
| DeepCTR | TensorFlow | 8K+ | 模型丰富，文档完善 |
| DeepCTR-Torch | PyTorch | 3.4K+ | PyTorch版本，易于扩展 |

### 6.2 最新开源

| 模型/框架 | 开源状态 | 核心价值 |
|-----------|---------|---------|
| DSAIN | 已开源 | 情境感知交互 |
| DCNv3 | 已开源 | 新一代交叉网络 |
| CASCADE | 已开源 | 因果推断CVR |
| TRACE | 已开源 | 延迟反馈建模 |
| MAC | 已开源 | 多归因学习 |

---

## 七、推荐研究方向

### 7.1 高优先级

1. **生成式排序（GRAB、CADET、HSTU）**
   - 趋势：统一推荐范式
   - 工业验证：百度、LinkedIn已落地

2. **LLM增强推荐系统**
   - 方向：知识蒸馏、特征增强
   - 潜力：解决冷启动和语义理解

3. **超长序列建模（SIM、TWIN、HyFormer、UxSID）**
   - 需求：用户行为数据爆炸式增长
   - 挑战：计算效率和效果平衡

4. **多任务学习优化（SMES、Hetero-MMoE）**
   - 工业价值：快手、Uber大规模验证
   - 趋势：异构专家、稀疏化

### 7.2 中优先级

5. **延迟反馈和去偏（TESLA、TRACE、ESCM2）**
   - 问题：真实场景核心痛点
   - 方向：因果推断方法

6. **多归因学习（MAL、MoAE）**
   - 创新：多目标归因建模
   - 效果：GAUC显著提升

7. **特征交叉新范式（DCNv3）**
   - 趋势：显式+隐式交叉融合
   - 突破：Self-Mot降噪机制

---

## 八、技术路线图

### 阶段一：基础奠基

```
LR → FM → DeepFM → DCN → DIN → DIEN
```

掌握经典模型原理，理解特征交叉、注意力机制、兴趣演化等核心概念。

### 阶段二：进阶提升

```
ESMM → MMoE → PLE → SIM → HSTU
```

学习多任务学习、超长序列建模、生成式排序等进阶技术。

### 阶段三：前沿探索

```
HyFormer/GRAB/CADET → ELEC/CTRL → TESLA/TRACE
```

跟踪最新研究进展，探索LLM增强、延迟反馈建模等前沿方向。

### 参考资源

- 论文检索：arXiv、KDD、RecSys、WWW、SIGIR
- 代码实现：GitHub开源项目
- 工程实践：大厂技术博客（腾讯、百度、快手、Meta）

---

> 本文档基于2024-2026年最新研究成果整理，持续更新中。
