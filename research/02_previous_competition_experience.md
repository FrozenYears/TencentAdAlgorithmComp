# 腾讯广告算法大赛 - 往届经验深度研究

> 基于2025年第十届腾讯广告算法大赛决赛答辩及技术分享整理

---

## 一、2025年比赛回顾

### 1.1 赛题：全模态生成式推荐

**核心任务**：基于用户全模态历史行为序列，预测下一个可能交互的广告。

**数据集规模**：
| 数据集 | 规模 | 用途 |
|--------|------|------|
| TencentGR-1M | 百万级用户 | 初赛/复赛 |
| TencentGR-10M | 千万级用户 | 决赛扩展 |

**赛题特点**：
- 多模态输入：文本、图像、视频等广告创意信息
- 生成式范式：从传统判别式推荐转向生成式推荐
- 行为序列建模：需要理解用户复杂的行为模式

### 1.2 冠军方案：Echoch团队

**团队成员**：华中科技大学、北京大学、中国科学技术大学联合团队

**核心创新**：

#### 三级会话体系与逐位置行为条件化建模
```python
# 概念示意：Per-position Action Conditioning
class ActionConditionedLayer:
    def __init__(self):
        self.film_layer = FiLM()  # 特征调制
        self.gate_layer = GatedFusion()  # 门控融合
        self.attn_bias = AttentionBias()  # 注意力偏置
    
    def forward(self, hidden, action_type):
        # 根据行为类型（点击/转化）动态调整表征
        return self.film_layer(hidden, action_type)
```

**关键技术组件**：

| 技术 | 作用 | 效果 |
|------|------|------|
| FiLM特征空间变换层 | 实现点击/转化预测的差异化表征 | 解决噪声淹没有效信号 |
| HSTU架构 + RoPE位置编码 | 高效序列建模 | 长序列处理能力提升 |
| RQ-KMeans语义ID | 残差量化生成语义Token | 长尾广告冷启动改善 |
| Random-k正则化 | 防止过拟合 | 泛化能力增强 |
| Muon优化器 | 专为稠密权重设计 | 显存减45%，收敛快40% |

**最终成绩**：
- 线上核心指标 **+0.013** 提升
- 获得 **200万元** 冠军奖金

### 1.3 亚军方案：leejt团队

**团队成员**：中山大学

**核心思路**：大数据、大模型，系统性解决数据脏乱差问题

**技术亮点**：

#### 共享词表+哈希编码
```python
# 处理超大规模ID空间
class SharedVocabHash:
    def __init__(self, vocab_size=1000000):
        self.hash_fn = xxhash.xxh64
        self.vocab_size = vocab_size
    
    def encode(self, item_id):
        # 哈希映射到共享词表
        return self.hash_fn(str(item_id)) % self.vocab_size
```

#### Session划分+异构时序图
- 将用户行为划分为不同Session
- 构建异构时序图捕捉复杂交互模式
- G-MLP/SASRec混合架构平衡效率与效果

### 1.4 技术创新奖：料峭春风吹酒醒团队

**团队成员**：中国科学院计算技术研究所

**核心创新**：统一生成式检索与排序的单模型范式

**技术栈**：
```
FlashAttention → 高效注意力计算
SwiGLU        → 激活函数优化
RMSNorm       → 稳定训练过程
RoPE          → 旋转位置编码
DeepSeek-V3 MoE → 混合专家系统
```

**创新点**：
- 将检索（Retrieval）和排序（Ranking）统一到单一生成式模型
- 避免传统两阶段范式的信息损失
- 端到端优化整体推荐效果

---

## 二、关键技术洞察

### 2.1 行为条件化（Action Conditioning）

**三机制协同**：

```
┌─────────────────────────────────────────────────┐
│           Action Conditioning Framework          │
├─────────────┬─────────────┬─────────────────────┤
│ FiLM调制    │ 门控融合    │ 注意力偏置          │
│ (特征变换)  │ (信息筛选)  │ (注意力引导)        │
└─────────────┴─────────────┴─────────────────────┘
```

**核心价值**：
- **FiLM特征调制**：根据行为类型动态调整特征空间
- **Gated Fusion门控融合**：自适应选择重要信息
- **Attention Biasing注意力偏置**：引导模型关注关键行为

**解决的问题**：彻底解决噪声淹没有效信号的问题，使模型能够区分"随意点击"和"真实兴趣"。

### 2.2 语义ID构建

#### RQ-KMeans算法

**原理**：残差量化K-Means，使语义相近广告共享Token前缀

```python
# RQ-KMeans 概念示意
class RQKMeans:
    def __init__(self, n_levels=3, n_clusters=256):
        self.n_levels = n_levels
        self.n_clusters = n_clusters
    
    def fit(self, embeddings):
        # 多层残差量化
        codes = []
        residual = embeddings
        for level in range(self.n_levels):
            kmeans = KMeans(self.n_clusters)
            codes.append(kmeans.fit_predict(residual))
            residual = residual - kmeans.cluster_centers_[codes[-1]]
        return codes
```

**优势**：
- 语义相似的广告共享ID前缀
- 提升长尾广告的冷启动效果
- 支持高效的近似最近邻搜索

### 2.3 优化器策略

| 优化器 | 应用场景 | 特点 |
|--------|----------|------|
| Muon | 稠密权重（Transformer层） | Newton-Schulz迭代正交化，显存减45% |
| AdamW | 稀疏Embedding层 | 传统自适应学习率，稳定可靠 |

**Muon优化器核心**：
```python
# Muon: Momentum + Orthogonalization
class MuonOptimizer:
    def step(self):
        for param in self.dense_params:
            grad = param.grad
            # Newton-Schulz迭代进行正交化
            orthogonalized = self.newton_schulz(grad)
            # 动量更新
            param.data -= lr * orthogonalized
```

### 2.4 推理优化

**推理解耦架构**：

```
┌──────────────────┐    ┌──────────────────┐
│   用户侧         │    │   广告侧         │
│   在线Transformer │    │   离线Faiss索引   │
│   (实时计算)      │    │   (预计算)        │
└────────┬─────────┘    └────────┬─────────┘
         │                       │
         └───────────┬───────────┘
                     ▼
            Large Negative Banks InfoNCE
```

**优化效果**：
- 用户侧只计算一次Transformer前向
- 广告侧通过Faiss索引高效检索
- Large Negative Banks扩大负样本池，提升对比学习效果

---

## 三、特征工程最佳实践

### 3.1 时间特征

**多粒度时间特征**：

| 粒度 | 特征示例 | 作用 |
|------|----------|------|
| 秒级 | 距上次点击间隔 | 捕捉即时兴趣 |
| 周级 | 星期几、工作日/周末 | 周期性行为模式 |
| 月级 | 月份、季节 | 长期兴趣变化 |

**时间衰减设计**：
```python
# 对数尺度时间分桶
def time_decay_bucket(delta_seconds):
    """将时间间隔映射到对数尺度桶"""
    if delta_seconds <= 0:
        return 0
    # 对数分桶：1min, 5min, 30min, 2h, 12h, 3d, 2周, ...
    bucket = int(math.log(delta_seconds / 60 + 1, 2))
    return min(bucket, 15)  # 最大15个桶
```

**设计原则**：区分随机点击和真实兴趣，近期行为权重更高。

### 3.2 会话特征

**三级会话体系**：

```
Request Session（请求级）
    └── 每次广告请求的上下文
        
Session（会话级）
    └── 用户一次使用APP的行为序列
        
Visit Session（访问级）
    └── 跨多次访问的长期行为模式
```

**核心思想**：模拟用户浏览和决策心理，不同层级的会话反映不同的用户意图。

### 3.3 多模态特征

**统一Embedding方案**：

```python
# 多模态大模型生成统一表征
class MultiModalEncoder:
    def __init__(self):
        self.text_encoder = TextEncoder()  # 文本
        self.image_encoder = ImageEncoder()  # 图像
        self.video_encoder = VideoEncoder()  # 视频
        self.fusion_layer = CrossModalFusion()
    
    def encode(self, ad):
        # 多模态融合
        text_emb = self.text_encoder(ad.title, ad.description)
        image_emb = self.image_encoder(ad.creative_image)
        return self.fusion_layer(text_emb, image_emb)
```

**特征处理原则**：
- **特征压缩**：高维ID特征降维，减少计算开销
- **特征选择**：舍弃低质量特征，避免引入噪声
- **统一表征**：多模态信息映射到同一语义空间

---

## 四、常见踩坑与注意事项

### 4.1 数据问题

#### 数据穿越问题
**现象**：训练时使用了未来信息，导致线上效果下降

**解决方案**：贝叶斯平滑
```python
# 贝叶斯平滑处理点击率
def bayesian_smoothing(clicks, impressions, alpha_prior, beta_prior):
    """避免使用全局统计造成数据穿越"""
    smooth_ctr = (clicks + alpha_prior) / (impressions + alpha_prior + beta_prior)
    return smooth_ctr
```

#### 数据质量问题
- **特征筛选**：剔除覆盖率过低或方差过小的特征
- **聚合统计**：注意时间窗口，避免信息泄漏

### 4.2 模型训练

#### 过拟合问题
- **验证集划分**：严格按时间划分，模拟线上场景
- **早停策略**：监控验证集指标，及时停止训练
- **正则化**：Dropout、Weight Decay、Random-k等

#### 内存优化
```python
# 减少Numpy/Pandas使用，改用流式处理
# 错误示范：
df = pd.read_csv("large_file.csv")  # 内存爆炸

# 正确示范：
for chunk in pd.read_csv("large_file.csv", chunksize=10000):
    process(chunk)
```

### 4.3 工程实现

#### 代码共享问题
- 比赛鼓励自主创新，避免过度依赖他人代码
- 理解原理比复制代码更重要

#### 特征处理
```python
# FFM格式转换示例
def convert_to_ffm_format(features, field_map):
    """将特征转换为FFM格式"""
    ffm_line = []
    for feat_name, feat_value in features.items():
        field_idx = field_map[feat_name]
        ffm_line.append(f"{field_idx}:{feat_name}:{feat_value}")
    return " ".join(ffm_line)
```

---

## 五、评委关注点

### 5.1 评估指标

**双指标体系**：

| 指标 | 含义 | 侧重点 |
|------|------|--------|
| HitRate@10 | 前10个推荐中命中比例 | 召回能力 |
| NDCG@10 | 前10个推荐的排序质量 | 排序精度 |

**复赛特殊规则**：
- 行为类型权重：**转化行为权重2.5倍**
- 体现广告场景对转化的重视

### 5.2 创新方向

评委重点关注的创新方向：

1. **生成式模型结构创新**
   - 从判别式到生成式的范式转变
   - 统一检索与排序的新架构

2. **多模态Embedding应用**
   - 如何有效融合多模态信息
   - 跨模态对齐与表征学习

3. **算法工程Co-design**
   - 算法创新与工程优化的结合
   - 推理效率与效果的平衡

---

## 六、关键资源

### 6.1 官方资源

| 资源 | 链接 | 说明 |
|------|------|------|
| 2025基线代码 | [github.com/TencentAdvertisingAlgorithmCompetition/baseline_2025](https://github.com/TencentAdvertisingAlgorithmCompetition/baseline_2025) | 官方提供的起始代码 |
| 数据集 | HuggingFace TAAC2025 | 比赛数据集下载 |

### 6.2 决赛方案代码

**开源代码库**：
- `kaipengm2/2025-tencent-advertising-algorithm-competition-finalist`
- 包含决赛选手的完整方案实现

### 6.3 学术论文

**推荐阅读**：
```
arXiv:2604.04976 - The Tencent Advertising Algorithm Challenge 2025
```

**论文内容**：
- 2025年赛题详细描述
- 决赛团队技术方案汇总
- 评估指标与实验结果分析

---

## 附录：关键技术术语表

| 术语 | 英文 | 含义 |
|------|------|------|
| FiLM | Feature-wise Linear Modulation | 特征级线性调制 |
| HSTU | Hierarchical Sequential Transduction Unit | 层级序列转换单元 |
| RoPE | Rotary Position Embedding | 旋转位置编码 |
| RQ-KMeans | Residual Quantization K-Means | 残差量化K均值 |
| Muon | Momentum + Orthogonalization | 动量正交优化器 |
| InfoNCE | Noise Contrastive Estimation | 噪声对比估计 |
| MoE | Mixture of Experts | 混合专家系统 |

---

> **研究建议**：基于2025年经验，重点关注生成式推荐范式、多模态特征融合、以及推理效率优化三个方向。冠军方案的行为条件化机制和语义ID构建方法值得深入研究和改进。

---

*文档版本：v1.0*  
*最后更新：2026年5月*  
*数据来源：2025年腾讯广告算法大赛决赛公开资料*