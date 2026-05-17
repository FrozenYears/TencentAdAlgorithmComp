# 广告系统工程优化 - 最佳实践

## 一、显存优化

### 1.1 混合精度训练

#### torch.cuda.amp 使用

PyTorch 自动混合精度（AMP）通过动态转换数据类型，在保持模型精度的同时减少显存占用和计算时间。

```python
import torch
from torch.cuda.amp import autocast, GradScaler

# 初始化梯度缩放器
scaler = GradScaler()

for data, target in dataloader:
    optimizer.zero_grad()
    
    # 前向传播使用混合精度
    with autocast():
        output = model(data)
        loss = criterion(output, target)
    
    # 反向传播使用缩放后的梯度
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

#### FP16/BF16 选择

| 数据类型 | 显存占用 | 数值稳定性 | 适用场景 |
|----------|----------|------------|----------|
| FP16 | 2字节 | 需要损失缩放 | 推理、部分训练场景 |
| BF16 | 2字节 | 更稳定 | 训练首选（A100/H100） |
| FP32 | 4字节 | 最稳定 | 精度敏感操作 |

```python
# BF16 训练配置
torch.set_float32_matmul_precision('high')
with autocast(dtype=torch.bfloat16):
    output = model(input)
```

#### 损失缩放策略

```python
# 静态损失缩放
scaler = GradScaler(init_scale=2**16)

# 动态损失缩放（推荐）
scaler = GradScaler(
    init_scale=2**16,
    growth_factor=2.0,
    backoff_factor=0.5,
    growth_interval=2000
)
```

### 1.2 梯度检查点

#### 时间换空间

梯度检查点用额外的前向计算换取显存节省，典型可减少 60-70% 显存。

```python
from torch.utils.checkpoint import checkpoint

class CheckpointedModel(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)
    
    def forward(self, x):
        for layer in self.layers:
            # 对每个层使用检查点
            x = checkpoint(layer, x, use_reentrant=False)
        return x
```

#### 选择性检查点

仅对计算密集但显存占用大的模块使用检查点：

```python
# 推荐检查点的模块
# 1. 大型线性层
# 2. 注意力机制
# 3. 卷积层

# 不推荐检查点的模块
# 1. 轻量级激活函数
# 2. 归一化层
# 3. Dropout
```

### 1.3 优化器选择

#### Muon 优化器：显存减 45%

Muon（MomentUm Orthogonalized by Newton-schulz）通过正交化动量更新，显著减少显存占用。

```python
# Muon 优化器配置
from muon import Muon

# 分组参数：Muon 处理 2D 参数，AdamW 处理其他参数
param_groups = [
    {'params': [p for n, p in model.named_parameters() if p.dim() >= 2]},
    {'params': [p for n, p in model.named_parameters() if p.dim() < 2]}
]

optimizer = Muon(param_groups, lr=0.02, momentum=0.95)
```

#### AdamW：稀疏参数优化

```python
# AdamW 针对稀疏嵌入的优化配置
optimizer = AdamW([
    {'params': embedding_params, 'lr': 1e-3, 'weight_decay': 0.01},
    {'params': dense_params, 'lr': 1e-4, 'weight_decay': 0.1}
])
```

#### 混合优化器策略

```python
# 2D+ 参数使用 Muon，其他使用 AdamW
class HybridOptimizer:
    def __init__(self, model, lr=0.02):
        muon_params = []
        adamw_params = []
        
        for name, param in model.named_parameters():
            if param.dim() >= 2:
                muon_params.append(param)
            else:
                adamw_params.append(param)
        
        self.muon_opt = Muon(muon_params, lr=lr)
        self.adamw_opt = AdamW(adamw_params, lr=lr * 0.1)
    
    def step(self):
        self.muon_opt.step()
        self.adamw_opt.step()
```

## 二、训练加速

### 2.1 数据加载

#### DataLoader 优化

```python
dataloader = DataLoader(
    dataset,
    batch_size=256,
    num_workers=8,              # CPU 核心数的 1/2 到 3/4
    pin_memory=True,            # 锁页内存，加速 CPU->GPU 传输
    persistent_workers=True,    # 保持 worker 进程存活
    prefetch_factor=2,          # 每个 worker 预取批次数量
    drop_last=True              # 避免最后不完整批次
)
```

#### 多进程加载

```python
# 自定义 Worker 初始化
def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

dataloader = DataLoader(
    dataset,
    num_workers=8,
    worker_init_fn=worker_init_fn
)
```

#### 预取策略

```python
# 双缓冲预取
class PrefetchLoader:
    def __init__(self, dataloader, device):
        self.dataloader = dataloader
        self.device = device
        self.stream = torch.cuda.Stream()
    
    def __iter__(self):
        loader_iter = iter(self.dataloader)
        first_batch = next(loader_iter)
        next_batch = self._async_copy(first_batch)
        
        for batch in loader_iter:
            current_batch = next_batch
            next_batch = self._async_copy(batch)
            yield current_batch
        yield next_batch
    
    def _async_copy(self, batch):
        with torch.cuda.stream(self.stream):
            return {k: v.to(self.device, non_blocking=True) 
                    for k, v in batch.items()}
```

### 2.2 模型并行

#### 数据并行（DDP）

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# 初始化进程组
dist.init_process_group(backend='nccl')
local_rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local_rank)

# 包装模型
model = DDP(model, device_ids=[local_rank])
```

#### 模型并行

```python
# 层间并行：不同层放不同 GPU
class ModelParallel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(100000, 512).cuda(0)
        self.transformer = nn.TransformerEncoder(...).cuda(1)
        self.classifier = nn.Linear(512, 2).cuda(1)
    
    def forward(self, x):
        x = self.embedding(x.cuda(0))
        x = x.cuda(1)
        x = self.transformer(x)
        return self.classifier(x)
```

#### 流水线并行

```python
from torch.distributed.pipeline.sync import Pipe

# 将模型分层
layers = nn.Sequential(
    nn.Linear(1000, 500),
    nn.ReLU(),
    nn.Linear(500, 100),
    nn.ReLU(),
    nn.Linear(100, 10)
)

# 流水线并行（4 个 micro-batch）
model = Pipe(nn.Sequential(layers), chunks=4)
```

### 2.3 编译优化

#### torch.compile

```python
# 基础编译
model = torch.compile(model)

# 详细配置
model = torch.compile(
    model,
    mode='reduce-overhead',     # 减少开销模式
    fullgraph=True,             # 强制完整图编译
    backend='inductor'          # 使用 Inductor 后端
)

# 动态形状支持
model = torch.compile(
    model,
    dynamic=True,               # 支持动态输入尺寸
    options={
        'triton.cudagraphs': True
    }
)
```

#### 静态计算图

```python
# 标记静态区域
@torch.compile(fullgraph=True)
def static_forward(x, weight):
    return torch.matmul(x, weight)

# 使用 torch.jit.trace 固定计算图
traced_model = torch.jit.trace(model, example_input)
```

#### 算子融合

```python
# 自动融合示例：LayerNorm + Linear
class FusedLayerNormLinear(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.linear = nn.Linear(d_model, d_ff)
    
    @torch.compile
    def forward(self, x):
        return self.linear(self.norm(x))
```

## 三、推理优化

### 3.1 模型压缩

#### 知识蒸馏

```python
class DistillationLoss(nn.Module):
    def __init__(self, temperature=4.0, alpha=0.7):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
    
    def forward(self, student_logits, teacher_logits, labels):
        # 软标签损失
        soft_loss = self.kl_loss(
            F.log_softmax(student_logits / self.temperature, dim=-1),
            F.softmax(teacher_logits / self.temperature, dim=-1)
        ) * (self.temperature ** 2)
        
        # 硬标签损失
        hard_loss = self.ce_loss(student_logits, labels)
        
        return self.alpha * soft_loss + (1 - self.alpha) * hard_loss
```

#### 量化（INT8/INT4）

```python
# 动态量化
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {nn.Linear, nn.LSTM},
    dtype=torch.qint8
)

# 静态量化（更优性能）
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
torch.quantization.prepare(model, inplace=True)
# 校准数据
for data in calibration_loader:
    model(data)
torch.quantization.convert(model, inplace=True)

# GPTQ INT4 量化（Transformer 模型）
from auto_gptq import AutoGPTQForCausalLM
quantized_model = AutoGPTQForCausalLM.from_quantized(
    model_path,
    device='cuda:0',
    use_triton=True
)
```

#### 剪枝

```python
import torch.nn.utils.prune as prune

# 非结构化剪枝
prune.l1_unstructured(module, name='weight', amount=0.3)

# 结构化剪枝（按通道）
prune.ln_structured(module, name='weight', amount=0.5, n=2, dim=0)

# 全局剪枝
parameters_to_prune = [(model.layer1, 'weight'), (model.layer2, 'weight')]
prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.2
)
```

### 3.2 向量检索

#### Faiss ANN 索引

```python
import faiss

# IVF 索引（倒排文件）
d = 768  # 向量维度
nlist = 1000  # 聚类中心数量
quantizer = faiss.IndexFlatL2(d)
index = faiss.IndexIVFFlat(quantizer, d, nlist)

# 训练索引
index.train(train_vectors)
index.add(database_vectors)

# 检索
k = 10
distances, indices = index.search(query_vectors, k)
```

#### HNSW 算法

```python
# HNSW 索引配置
index = faiss.IndexHNSWFlat(d, 32)  # 32 为连接数
index.hnsw.efSearch = 128           # 搜索时的动态候选列表大小
index.hnsw.efConstruction = 200     # 构建时的候选列表大小

# 添加数据
index.add(database_vectors)
distances, indices = index.search(query_vectors, k)
```

#### 量化压缩

```python
# Product Quantization (PQ)
m = 32  # 子量化器数量
nbits = 8  # 每个子量化器的比特数
index = faiss.IndexPQ(d, m, nbits)

# Optimized Product Quantization (OPQ)
index = faiss.IndexOPQ(d, m)
index.train(train_vectors)
index.add(database_vectors)

# Scalar Quantizer
index = faiss.IndexScalarQuantizer(d, faiss.ScalarQuantizer.QT_8bit)
```

### 3.3 推理解耦

#### 用户侧：在线 Transformer

```python
# 用户特征实时编码
class UserEncoder:
    def __init__(self, model_path):
        self.model = load_model(model_path)
        self.cache = LRUCache(maxsize=100000)
    
    def encode(self, user_id, user_features):
        # 缓存命中检查
        if user_id in self.cache:
            return self.cache[user_id]
        
        # 实时推理
        with torch.no_grad():
            embedding = self.model(user_features)
        
        self.cache[user_id] = embedding
        return embedding
```

#### 广告侧：离线预计算

```python
# 广告向量离线批处理
def batch_compute_ad_embeddings(ad_list, model, batch_size=1024):
    embeddings = []
    for i in range(0, len(ad_list), batch_size):
        batch = ad_list[i:i+batch_size]
        with torch.no_grad():
            batch_emb = model.encode_ad(batch)
        embeddings.append(batch_emb.cpu())
    return torch.cat(embeddings)

# 存储到向量数据库
ad_embeddings = batch_compute_ad_embeddings(ads, model)
index.add(ad_embeddings.numpy())
```

#### 缓存策略

```python
from functools import lru_cache
import redis

class InferenceCache:
    def __init__(self, redis_client, ttl=3600):
        self.redis = redis_client
        self.ttl = ttl
    
    def get_or_compute(self, key, compute_fn):
        cached = self.redis.get(key)
        if cached:
            return pickle.loads(cached)
        
        result = compute_fn()
        self.redis.setex(key, self.ttl, pickle.dumps(result))
        return result

# 分层缓存：本地缓存 -> Redis -> 计算
@lru_cache(maxsize=10000)
def local_compute(user_id):
    return cache.get_or_compute(
        f"user:{user_id}",
        lambda: compute_user_embedding(user_id)
    )
```

## 四、特征工程优化

### 4.1 特征存储

#### 特征服务架构

```python
# 特征服务接口
class FeatureStore:
    def __init__(self):
        self.online_store = RedisClient()
        self.offline_store = ParquetStore()
    
    def get_online_features(self, entity_id, feature_names):
        """在线特征获取（低延迟）"""
        pipe = self.online_store.pipeline()
        for fname in feature_names:
            pipe.hget(f"entity:{entity_id}", fname)
        return dict(zip(feature_names, pipe.execute()))
    
    def get_offline_features(self, entity_ids, feature_names):
        """离线特征获取（高吞吐）"""
        return self.offline_store.query(entity_ids, feature_names)
```

#### 在线/离线特征

```python
# 特征定义
FEATURE_CONFIG = {
    # 在线特征：实时更新
    'online': {
        'user_click_count_1h': {'ttl': 3600, 'update': 'realtime'},
        'ad_ctr_1d': {'ttl': 86400, 'update': 'realtime'},
    },
    # 离线特征：定时更新
    'offline': {
        'user_age': {'ttl': 86400 * 30, 'update': 'daily'},
        'ad_category': {'ttl': 86400, 'update': 'daily'},
    }
}
```

#### 特征缓存

```python
class MultiLevelCache:
    def __init__(self):
        self.l1_cache = {}  # 进程内缓存
        self.l2_cache = RedisClient()  # Redis 缓存
    
    def get(self, key):
        # L1 缓存
        if key in self.l1_cache:
            return self.l1_cache[key]
        
        # L2 缓存
        value = self.l2_cache.get(key)
        if value:
            self.l1_cache[key] = value
            return value
        
        return None
```

### 4.2 特征计算

#### 实时特征

```python
class RealtimeFeatureComputer:
    def __init__(self, kafka_consumer):
        self.consumer = kafka_consumer
        self.window_aggs = {}
    
    def process_event(self, event):
        """处理实时事件，更新滑动窗口聚合"""
        user_id = event['user_id']
        timestamp = event['timestamp']
        
        # 更新滑动窗口
        for window in ['1h', '6h', '24h']:
            key = f"{user_id}:{window}"
            self._update_window(key, timestamp, event)
    
    def _update_window(self, key, timestamp, event):
        """滑动窗口聚合"""
        if key not in self.window_aggs:
            self.window_aggs[key] = SlidingWindow(window_size=3600)
        self.window_aggs[key].add(timestamp, event)
```

#### 批量计算

```python
# 使用 Spark 进行批量特征计算
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("feature_computation").getOrCreate()

# 读取原始数据
raw_df = spark.read.parquet("hdfs://data/events/")

# 计算聚合特征
feature_df = raw_df.groupBy("user_id").agg(
    F.count("*").alias("total_clicks"),
    F.sum("revenue").alias("total_revenue"),
    F.avg("dwell_time").alias("avg_dwell_time")
)

# 写入特征存储
feature_df.write.mode("overwrite").parquet("hdfs://features/user_agg/")
```

#### 增量更新

```python
class IncrementalFeatureUpdater:
    def __init__(self, feature_store):
        self.feature_store = feature_store
    
    def update(self, delta_df):
        """增量更新特征"""
        for row in delta_df.collect():
            entity_id = row['entity_id']
            features = {k: v for k, v in row.asDict().items() 
                       if k != 'entity_id'}
            
            # 合并更新
            existing = self.feature_store.get(entity_id) or {}
            existing.update(features)
            self.feature_store.set(entity_id, existing)
```

## 五、数据处理优化

### 5.1 大规模数据加载

#### Parquet 格式优化

```python
import pyarrow.parquet as pq

# 读取优化：只加载需要的列
table = pq.read_table(
    'data.parquet',
    columns=['user_id', 'ad_id', 'click', 'timestamp']
)

# 使用谓词下推过滤
table = pq.read_table(
    'data.parquet',
    filters=[('date', '>=', '2024-01-01'), ('date', '<=', '2024-01-31')]
)
```

#### 列式存储

```python
# Pandas 读取优化
import pandas as pd

# 指定数据类型减少内存
dtypes = {
    'user_id': 'int32',
    'ad_id': 'int32',
    'click': 'int8',
    'timestamp': 'int64'
}

df = pd.read_parquet(
    'data.parquet',
    columns=list(dtypes.keys()),
    filters=[('partition_date', '=', '2024-01-01')]
)
```

#### 分区策略

```python
# 按日期分区写入
df.to_parquet(
    'data/events/',
    partition_cols=['date', 'hour'],
    engine='pyarrow',
    compression='snappy'
)

# Hive 风格分区读取
dataset = pq.ParquetDataset(
    'data/events/',
    filters=[('date', '=', '2024-01-01')]
)
```

### 5.2 内存优化

#### 减少 Numpy/Pandas 使用

```python
# 不推荐：大量小数组
for i in range(n):
    arr = np.array([data[i]])
    result.append(process(arr))

# 推荐：批量处理
batch_data = np.array(data)
results = process_batch(batch_data)
```

#### 原生数据结构

```python
# 使用 array 代替 list 存储数值
from array import array

# 不推荐
values = []
for v in data:
    values.append(v)

# 推荐
values = array('f')  # float32
for v in data:
    values.append(v)

# 使用 deque 处理滑动窗口
from collections import deque
window = deque(maxlen=1000)
```

#### 流式处理

```python
# 流式读取大文件
def stream_read_parquet(file_path, batch_size=10000):
    parquet_file = pq.ParquetFile(file_path)
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        yield batch.to_pandas()

# 内存友好的聚合
total = 0
count = 0
for batch in stream_read_parquet('large_file.parquet'):
    total += batch['value'].sum()
    count += len(batch)
average = total / count
```

## 六、分布式训练

### 6.1 PyTorch 分布式

#### DDP 使用

```python
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def train(rank, world_size):
    setup(rank, world_size)
    
    model = MyModel().to(rank)
    model = DDP(model, device_ids=[rank])
    
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=32)
    
    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        for batch in dataloader:
            loss = model(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

# 启动
torch.multiprocessing.spawn(train, args=(world_size,), nprocs=world_size)
```

#### 梯度同步

```python
# 手动梯度同步（梯度累积场景）
model = DDP(model, find_unused_parameters=False)

for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# 异步梯度同步
model = DDP(model, gradient_as_bucket_view=True)
```

#### 通信优化

```python
# 使用 NCCL 后端（GPU 间通信最优）
dist.init_process_group('nccl')

# 梯度压缩
from torch.distributed.algorithms.ddp_comm_hooks import (
    powerSGD_hook as powerSGD
)

state = powerSGD.PowerSGDState(
    process_group=None,
    matrix_approximation_rank=1,
    start_powerSGD_iter=10
)
model.register_comm_hook(state, powerSGD.powerSGD_hook)

# 混合精度通信
model = DDP(model, gradient_as_bucket_view=True)
```

### 6.2 多 GPU 训练

#### GPU 利用率优化

```python
# 监控 GPU 使用
def log_gpu_usage():
    for i in range(torch.cuda.device_count()):
        usage = torch.cuda.memory_allocated(i) / 1024**3
        total = torch.cuda.get_device_properties(i).total_mem / 1024**3
        print(f"GPU {i}: {usage:.2f}/{total:.2f} GB")

# 自动调整 batch size
def find_optimal_batch_size(model, sample_input, max_memory=0.9):
    batch_size = 1
    while True:
        try:
            input_batch = sample_input.repeat(batch_size, 1, 1)
            output = model(input_batch)
            loss = output.sum()
            loss.backward()
            
            if torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() > max_memory:
                return batch_size // 2
            batch_size *= 2
        except RuntimeError:  # OOM
            return batch_size // 2
```

#### 显存管理

```python
# 定期清理显存
def cleanup_memory():
    torch.cuda.empty_cache()
    gc.collect()

# 梯度累积减少显存峰值
optimizer.zero_grad(set_to_none=True)  # 更高效

# 使用 gradient checkpointing
from torch.utils.checkpoint import checkpoint
model = checkpoint_sequential(model, segments=4)
```

#### 负载均衡

```python
# 按计算量分配数据
class BalancedSampler:
    def __init__(self, dataset, num_replicas, rank):
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
    
    def __iter__(self):
        # 按样本复杂度排序
        complexities = [self.dataset.get_complexity(i) 
                       for i in range(len(self.dataset))]
        sorted_indices = np.argsort(complexities)
        
        # 交错分配确保负载均衡
        return iter(sorted_indices[self.rank::self.num_replicas])
```

## 七、监控与调试

### 7.1 训练监控

#### Loss 曲线

```python
import wandb

# 初始化 wandb
wandb.init(project="ad-system", config=hyperparameters)

# 记录训练指标
for step, batch in enumerate(dataloader):
    loss = train_step(batch)
    
    if step % 100 == 0:
        wandb.log({
            'train/loss': loss,
            'train/learning_rate': scheduler.get_lr(),
            'train/step': step
        })
```

#### 梯度分布

```python
def log_gradient_stats(model, step):
    """记录梯度统计信息"""
    grad_norms = []
    grad_maxs = []
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad.data
            grad_norms.append(grad.norm().item())
            grad_maxs.append(grad.abs().max().item())
    
    wandb.log({
        'grad/mean_norm': np.mean(grad_norms),
        'grad/max_norm': np.max(grad_norms),
        'grad/max_grad': np.max(grad_maxs),
        'step': step
    })
    
    # 梯度消失/爆炸检测
    if np.max(grad_norms) > 100:
        print(f"Warning: Gradient explosion at step {step}")
    if np.min(grad_norms) < 1e-7:
        print(f"Warning: Gradient vanishing at step {step}")
```

#### 显存使用

```python
class MemoryTracker:
    def __init__(self):
        self.history = []
    
    def log(self, step):
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        
        self.history.append({
            'step': step,
            'allocated_gb': allocated,
            'reserved_gb': reserved
        })
        
        wandb.log({
            'memory/allocated_gb': allocated,
            'memory/reserved_gb': reserved,
            'step': step
        })
    
    def detect_leak(self, threshold=0.1):
        """检测显存泄漏"""
        if len(self.history) > 100:
            recent = self.history[-100:]
            growth = recent[-1]['allocated_gb'] - recent[0]['allocated_gb']
            if growth > threshold:
                print(f"Potential memory leak: {growth:.2f}GB growth")
```

### 7.2 性能分析

#### Profiler 使用

```python
from torch.profiler import profile, record_function, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(
        wait=1,
        warmup=1,
        active=3,
        repeat=2
    ),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    for step, batch in enumerate(dataloader):
        if step >= 10:
            break
        with record_function("data_loading"):
            data = batch.to(device)
        with record_function("forward"):
            output = model(data)
        with record_function("backward"):
            loss.backward()
        prof.step()

# 打印统计
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
```

#### 瓶颈识别

```python
def analyze_bottleneck(prof):
    """分析训练瓶颈"""
    key_averages = prof.key_averages()
    
    # 计算各阶段耗时
    data_time = sum(k.cuda_time for k in key_averages if 'data' in k.key)
    compute_time = sum(k.cuda_time for k in key_averages if 'forward' in k.key or 'backward' in k.key)
    comm_time = sum(k.cuda_time for k in key_averages if 'nccl' in k.key.lower())
    
    total = data_time + compute_time + comm_time
    
    print(f"数据加载: {data_time/total*100:.1f}%")
    print(f"计算: {compute_time/total*100:.1f}%")
    print(f"通信: {comm_time/total*100:.1f}%")
    
    # 瓶颈判断
    if data_time / total > 0.3:
        print("瓶颈: 数据加载 - 增加 num_workers 或使用 prefetch")
    if comm_time / total > 0.2:
        print("瓶颈: 通信 - 使用梯度压缩或异步通信")
```

#### 优化建议

```python
def get_optimization_suggestions(prof):
    """根据 profile 结果给出优化建议"""
    suggestions = []
    
    # 检查 GPU 利用率
    gpu_util = prof.key_averages()
    cpu_time = sum(k.cpu_time for k in gpu_util)
    gpu_time = sum(k.cuda_time for k in gpu_util)
    
    if gpu_time / cpu_time < 0.5:
        suggestions.append("GPU 利用率低，考虑增加 batch_size 或使用 CUDA Graph")
    
    # 检查显存碎片
    if torch.cuda.memory_stats()['num_alloc_retries'] > 0:
        suggestions.append("显存碎片化，考虑使用 memory pooling")
    
    # 检查小 kernel
    small_kernels = [k for k in gpu_util if k.cuda_time < 100]
    if len(small_kernels) > 100:
        suggestions.append("大量小 kernel，考虑使用 torch.compile 融合算子")
    
    return suggestions
```

## 八、实用工具

### 8.1 PyTorch 生态

#### torch.compile

```python
# 基础用法
model = torch.compile(model)

# 模式选择
model = torch.compile(model, mode='default')      # 平衡模式
model = torch.compile(model, mode='reduce-overhead') # 减少开销
model = torch.compile(model, mode='max-autotune')   # 最大性能

# 调试编译问题
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.verbose = True
```

#### torch.cuda.amp

```python
# 自动混合精度训练
scaler = torch.cuda.amp.GradScaler()

with torch.cuda.amp.autocast():
    output = model(input)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()

# 自定义精度
with torch.cuda.amp.autocast(dtype=torch.bfloat16):
    output = model(input)
```

#### torch.distributed

```python
# 启动分布式训练
torchrun --nproc_per_node=4 --master_port=29500 train.py

# 获取分布式信息
local_rank = int(os.environ['LOCAL_RANK'])
world_size = int(os.environ['WORLD_SIZE'])

# 通信原语
dist.barrier()  # 同步所有进程
dist.all_reduce(tensor)  # 全归约
dist.broadcast(tensor, src=0)  # 广播
```

### 8.2 第三方工具

#### DeepSpeed

```python
import deepspeed

# ZeRO 优化配置
ds_config = {
    "train_batch_size": 256,
    "gradient_accumulation_steps": 4,
    "fp16": {"enabled": True},
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": True
    }
}

model, optimizer, _, _ = deepspeed.initialize(
    model=model,
    config=ds_config
)
```

#### FairScale

```python
from fairscale.nn.data_parallel import ShardedDataParallel as ShardedDDP
from fairscale.optim.oss import OSS

# 分片优化器
optimizer = OSS(params=model.parameters(), optim=torch.Adam, lr=1e-3)

# 分片数据并行
model = ShardedDDP(model, optimizer)

# 梯度检查点
from fairscale.nn.checkpoint import checkpoint_wrapper
model = checkpoint_wrapper(model)
```

#### Megatron-LM

```python
# Megatron-LM 并行训练配置
from megatron import get_args
from megatron.model import GPTModel
from megatron.training import pretrain

# 模型并行配置
def model_provider(pre_process=True, post_process=True):
    model = GPTModel(
        num_layers=32,
        hidden_size=4096,
        num_attention_heads=32,
        max_position_embeddings=2048,
        parallel_output=True
    )
    return model

# 启动训练
pretrain(
    model_provider,
    forward_step,
    train_valid_test_datasets_provider
)
```

---

## 附录：快速参考表

| 优化项 | 方法 | 预期收益 |
|--------|------|----------|
| 显存优化 | BF16 + 梯度检查点 | 减少 60-70% 显存 |
| 训练加速 | torch.compile | 提速 20-50% |
| 推理优化 | INT8 量化 | 提速 2-4x |
| 数据加载 | Parquet + 多进程 | 提速 3-5x |
| 分布式训练 | DDP + 梯度累积 | 线性扩展 |
| 向量检索 | HNSW + PQ | 毫秒级检索 |
