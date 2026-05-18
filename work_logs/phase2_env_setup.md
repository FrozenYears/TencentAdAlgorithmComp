# 阶段二：环境搭建与数据准备 - 工作日志（更新）

## 阶段信息
- **阶段名称**: 环境搭建与数据准备
- **开始时间**: 2026年5月17日
- **当前状态**: 进行中

## 已完成任务

### 1. Python虚拟环境创建
**状态**: ✅ 完成

### 2. PyTorch安装
**状态**: ✅ 完成

**配置**:
- PyTorch: 2.11.0+cu128
- CUDA: 12.8
- GPU: NVIDIA GeForce RTX 5070 Laptop GPU (8GB)

### 3. 依赖库安装
**状态**: ✅ 完成

### 4. 数据集下载
**状态**: ✅ 完成

**下载的数据**:
- user_feat: 1,001,845条
- item_feat: 4,783,154条
- candidate: 660,000条
- seq: 1,001,845条
- mm_emb_81_32: 4,742,961条
- indexer.pkl

### 5. Baseline代码克隆
**状态**: ✅ 完成

### 6. 数据格式修复
**状态**: ✅ 完成

**修改的文件**:
- `baseline/dataset.py`: 修改null值处理逻辑，使用numpy的where+is_valid替代pandas

### 7. 数据加载性能优化
**状态**: ✅ 完成

**优化前**: 使用pandas的fillna，加载缓慢
**优化后**: 使用numpy的where+is_valid，加载快速

**性能对比**:
- 序列数据(seq): 9秒
- 物品特征(item_feat): 17秒
- 用户特征(user_feat): 2秒
- 多模态嵌入(mm_emb): 96秒
- **总计**: 约2分钟

## 进行中任务

### 8. Baseline训练
**状态**: 🔄 进行中

**配置**:
- batch_size: 256
- hidden_units: 32
- num_blocks: 1
- num_heads: 1
- num_epochs: 1
- save_every_steps: 200

**当前状态**: 
- 数据加载完成
- 模型训练进行中
- 后台进程运行中

## 问题与解决

### 问题1: PyTorch CUDA不兼容
**现象**: RTX 5070 Laptop GPU的CUDA能力sm_120不被PyTorch 2.5.1支持
**解决**: 升级到PyTorch 2.11.0+cu128

### 问题2: 数据格式不匹配
**现象**: baseline期望parquet格式，HuggingFace保存的是Arrow格式
**解决**: 下载原始parquet文件

### 问题3: null值处理
**现象**: action_type列存在null值导致转换失败
**解决**: 修改dataset.py使用numpy的where+is_valid处理null值

### 问题4: 数据加载缓慢
**现象**: 使用pandas的fillna加载数据非常慢
**解决**: 使用numpy的where+is_valid替代pandas，性能提升显著

## 系统资源评估

**硬件配置**:
- CPU: AMD Ryzen 9 8945HX (8核16线程)
- RAM: 32GB
- GPU: RTX 5070 Laptop 8GB

**训练配置**:
- batch_size: 256 (适配8GB显存)
- num_workers: 4 (避免内存溢出)
- hidden_units: 32 (baseline默认)

## 下一步计划

1. 等待baseline训练完成
2. 生成提交文件
3. 第一次提交到官网
4. 开始特征工程

---

*日志更新时间: 2026年5月18日*
*日志维护人: Sisyphus*
