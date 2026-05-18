# CVR预测模型优化计划 - 目标AUC >= 0.7

## 一、当前问题分析

### 1.1 Baseline现状
- **模型**: SimpleCVRModel (GRU + Embedding + MLP)
- **数据**: 1000条demo样本
- **性能**: Train AUC 0.9789, Val AUC 0.5641
- **问题**: 严重过拟合（差距0.41）

### 1.2 过拟合原因
1. **数据量太少**: 1000样本无法支撑复杂模型
2. **模型过于复杂**: 参数量相对于数据量过大
3. **缺乏正则化**: 没有Dropout、L2正则等
4. **特征处理粗糙**: LabelEncoder不适用于高基数特征

## 二、优化策略（基于Oracle建议和研究结果）

### 2.1 核心策略：多任务学习（ESMM）
**原理**: CTCVR = CTR × CVR，利用CTR任务辅助CVR学习

**优势**:
- 解决样本选择偏差（SSB）
- 缓解数据稀疏性（DS）
- 利用CTR样本丰富CVR学习

**预期收益**: Val AUC提升3-5%

### 2.2 序列建模优化（DIN Target Attention）
**原理**: 用候选Item作为Query，用户历史行为作为Key/Value

**优势**:
- 自适应关注相关历史行为
- 比GRU更适合稀疏数据
- 可解释性强

**预期收益**: Val AUC提升2-3%

### 2.3 特征工程优化
1. **特征哈希**: 高基数特征映射到低维空间
2. **统计特征**: 用户/物品历史转化率
3. **交叉特征**: 用户×物品类目偏好
4. **时间特征**: 小时、星期、是否节假日

**预期收益**: Val AUC提升1-2%

### 2.4 正则化强化
1. **Embedding L2正则**: weight_decay=1e-5
2. **Dropout**: 0.3-0.5
3. **Early Stopping**: 监控Val AUC
4. **梯度裁剪**: max_norm=1.0

**预期收益**: Val AUC提升1%

## 三、详细工作计划

### 阶段1：数据增强与特征工程（Day 1）

#### 任务1.1：下载更多数据
- 从HuggingFace下载完整训练集
- 目标：10万+样本
- 验证：数据分布与demo一致

#### 任务1.2：特征工程
- 处理标量特征：LabelEncoder + Hash Trick
- 处理序列特征：统计长度、最近行为
- 处理稠密特征：归一化
- 生成统计特征：用户/物品历史CTR/CVR

#### 任务1.3：数据增强
- 负采样：平衡正负样本
- 序列截断：保留最近20-50个行为
- 特征Mask：随机遮蔽部分特征

### 阶段2：模型架构优化（Day 2）

#### 任务2.1：实现ESMM多任务学习
```python
class ESMM(nn.Module):
    def __init__(self, ...):
        self.embedding = EmbeddingLayer(...)
        self.ctr_tower = MLP(...)
        self.cvr_tower = MLP(...)
    
    def forward(self, x):
        shared_emb = self.embedding(x)
        ctr_pred = self.ctr_tower(shared_emb)
        cvr_pred = self.cvr_tower(shared_emb)
        ctcvr_pred = ctr_pred * cvr_pred
        return ctr_pred, ctcvr_pred
```

#### 任务2.2：实现DIN Target Attention
```python
class TargetAttention(nn.Module):
    def forward(self, query, keys, mask):
        # query: [B, D] - 候选Item
        # keys: [B, T, D] - 历史行为序列
        # 计算注意力分数
        att_input = torch.cat([query, keys, query-keys, query*keys], dim=-1)
        att_weight = self.mlp(att_input)  # [B, T, 1]
        att_weight = att_weight.masked_fill(~mask, -1e9)
        att_weight = torch.softmax(att_weight, dim=1)
        # 加权求和
        output = (att_weight * keys).sum(dim=1)
        return output
```

#### 任务2.3：实现DeepFM特征交叉
```python
class DeepFM(nn.Module):
    def __init__(self, ...):
        self.fm = FM()  # 二阶交叉
        self.dnn = MLP(...)  # 高阶交叉
    
    def forward(self, x):
        y_fm = self.fm(x)  # 显式交叉
        y_dnn = self.dnn(x)  # 隐式交叉
        return y_fm + y_dnn
```

### 阶段3：训练策略优化（Day 3）

#### 任务3.1：多任务损失函数
```python
def esmm_loss(ctr_pred, ctcvr_pred, ctr_label, cvr_label):
    ctr_loss = F.binary_cross_entropy(ctr_pred, ctr_label)
    ctcvr_loss = F.binary_cross_entropy(ctcvr_pred, cvr_label)
    return ctr_loss + ctcvr_loss
```

#### 任务3.2：学习率调度
- 初始学习率：1e-3
- Warmup：前10%步骤
- 余弦退火：逐步降低

#### 任务3.3：正则化策略
- Embedding L2正则：1e-5
- Dropout：0.3
- 梯度裁剪：max_norm=1.0
- Early Stopping：patience=5

### 阶段4：评估与调优（Day 4）

#### 任务4.1：离线评估
- 5折交叉验证
- AUC、LogLoss、GAUC
- 特征重要性分析

#### 任务4.2：超参数调优
- Embedding维度：16, 32, 64
- 隐藏层维度：64, 128, 256
- 学习率：1e-4, 5e-4, 1e-3
- Dropout：0.2, 0.3, 0.5

#### 任务4.3：消融实验
- ESMM vs 单任务CVR
- DIN vs GRU vs 无序列建模
- DeepFM vs MLP

## 四、预期效果

### 4.1 性能预期
| 优化阶段 | 预期AUC | 提升 |
|----------|---------|------|
| Baseline | 0.56 | - |
| +特征工程 | 0.62 | +6% |
| +ESMM | 0.67 | +5% |
| +DIN | 0.70 | +3% |
| +正则化 | 0.72 | +2% |

### 4.2 时间预期
- 阶段1：1天
- 阶段2：1天
- 阶段3：1天
- 阶段4：1天
- **总计：4天达到AUC 0.7+**

## 五、风险与应对

### 5.1 数据风险
- **风险**: 正式比赛数据格式可能不同
- **应对**: 设计灵活的数据处理pipeline

### 5.2 过拟合风险
- **风险**: 模型在小数据集上仍过拟合
- **应对**: 强正则化 + Early Stopping + 模型平均

### 5.3 计算资源风险
- **风险**: GPU内存不足
- **应对**: 减小batch_size，使用梯度累积

## 六、关键代码参考

### 6.1 DIN Target Attention (from torch-rechub)
```python
class ActivationUnit(nn.Module):
    def __init__(self, emb_dim, dims=[36]):
        super().__init__()
        self.attention = MLP(4 * emb_dim, dims=dims, activation='dice')
    
    def forward(self, history, target):
        seq_length = history.size(1)
        target = target.unsqueeze(1).expand(-1, seq_length, -1)
        att_input = torch.cat([target, history, target - history, target * history], dim=-1)
        att_weight = self.attention(att_input.view(-1, 4 * self.emb_dim))
        att_weight = att_weight.view(-1, seq_length)
        output = (att_weight.unsqueeze(-1) * history).sum(dim=1)
        return output
```

### 6.2 ESMM (from NextRec)
```python
class ESMM(nn.Module):
    def __init__(self, ...):
        self.embedding = EmbeddingLayer(features=self.all_features)
        self.ctr_tower = MLP(input_dim, output_dim=1, ...)
        self.cvr_tower = MLP(input_dim, output_dim=1, ...)
    
    def forward(self, x):
        input_flat = self.embedding(x, squeeze_dim=True)
        ctr_logit = self.ctr_tower(input_flat)
        cvr_logit = self.cvr_tower(input_flat)
        logits = torch.cat([ctr_logit, cvr_logit], dim=1)
        return logits
```

---

*计划制定日期: 2026年5月18日*
*目标: AUC >= 0.7*
*预计完成时间: 4天*
