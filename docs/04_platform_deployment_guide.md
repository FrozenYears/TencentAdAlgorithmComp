# 腾讯Angel平台部署指南

## 一、平台概述

### 1.1 硬件资源

| 资源 | 规格 |
|------|------|
| GPU | 20%单GPU，19GiB显存 |
| CPU | 9核 |
| 内存 | 55GiB |
| 缓存 | 20GB (USER_CACHE_PATH) |

### 1.2 软件环境

| 软件 | 版本 |
|------|------|
| 系统 | Ubuntu 22.04 |
| CUDA | 12.6 |
| Python | 3.10.20 |
| PyTorch | 2.7.1+cu126 |
| conda | 26.1.1 |

### 1.3 提交限制

| 限制 | 值 |
|------|------|
| 每日提交次数 | 学术3次/工业4次 |
| 推理时间限制 | 30分钟 |
| 脚本大小限制 | 100MB |

---

## 二、训练任务部署

### 2.1 文件准备

需要上传的文件：
```
deploy/
├── run.sh              # 训练入口脚本（必须）
├── train_v2.py         # 训练脚本
├── feature_engineering.py
├── esmm_din_model.py
├── evaluate.py
└── requirements.txt    # 依赖列表
```

### 2.2 环境变量

| 变量 | 说明 | 用途 |
|------|------|------|
| TRAIN_DATA_PATH | 训练数据路径 | 读取训练数据 |
| TRAIN_CKPT_PATH | 模型保存路径 | 保存checkpoint |
| TRAIN_TF_EVENTS_PATH | TensorBoard路径 | 记录训练指标 |
| USER_CACHE_PATH | 用户缓存路径 | 临时文件 |

### 2.3 Checkpoint命名规范

**必须**以`global_step`开头：
```
global_step20.lr=0.001.layer=2.head=1.hidden=128
```

允许的字符：`a-z A-Z 0-9 _ - = .`

### 2.4 提交步骤

1. 登录平台
2. 点击"Model Training"
3. 填写"Job Name"和"Job Description"
4. 点击"Local Upload"上传脚本
5. 点击"Submit"保存
6. 点击"Run"开始训练

---

## 三、模型评估部署

### 3.1 文件准备

需要上传的文件：
```
deploy/
├── infer.py            # 推理脚本（必须，包含main()函数）
├── prepare.sh          # 依赖安装脚本（可选）
├── feature_engineering.py
├── esmm_din_model.py
└── evaluate.py
```

### 3.2 环境变量

| 变量 | 说明 | 用途 |
|------|------|------|
| EVAL_DATA_PATH | 测试数据路径 | 读取测试数据 |
| EVAL_RESULT_PATH | 结果输出路径 | 保存predictions.json |
| MODEL_OUTPUT_PATH | 模型路径 | 加载训练好的模型 |
| EVAL_INFER_PATH | 脚本路径 | 读取上传的脚本 |
| USER_CACHE_PATH | 用户缓存路径 | 临时文件 |

### 3.3 输出格式

**必须**生成`predictions.json`：
```json
{
    "predictions": {
        "user_001": 0.8732,
        "user_002": 0.1245,
        "user_003": 0.5621
    }
}
```

**关键要求**：
- 每个key必须是测试集中的user_id（字符串）
- 每个value是预测的转化概率（0-1浮点数）
- 不能遗漏或多余user_id

### 3.4 提交步骤

1. 登录平台
2. 点击"Model Management"
3. 选择训练好的模型
4. 点击"Model Evaluation"
5. 上传推理脚本
6. 可选：上传prepare.sh安装依赖
7. 点击"Submit"

---

## 四、评估状态

| 状态 | 说明 |
|------|------|
| Pending | 任务已提交，排队中 |
| Waiting for Inference Resources | 等待计算资源 |
| Inference Running | 推理脚本执行中 |
| Waiting for Evaluation Resources | 推理完成，等待评分 |
| Evaluation Running | 平台评分中 |
| Success | 评估完成，可查看分数 |
| Failed | 评估失败，检查日志 |

---

## 五、依赖安装

### 5.1 使用prepare.sh

```bash
#!/bin/bash
# 安装额外依赖
pip install -q scikit-learn tqdm lightgbm

# 或使用conda
# conda install -y pandas
```

### 5.2 预装包

平台已预装：
- torch 2.7.1+cu126
- pandas 2.3.3
- numpy 2.2.5
- scikit-learn 1.7.2
- pyarrow 23.0.1
- datasets 2.14.7
- transformers 4.35.0

---

## 六、完整部署流程

### 6.1 训练阶段

```bash
# 1. 准备训练脚本
mkdir deploy
cp src/feature_engineering.py deploy/
cp src/esmm_din_model.py deploy/
cp src/evaluate.py deploy/
cp src/train_v2.py deploy/

# 2. 创建run.sh
cat > deploy/run.sh << 'EOF'
#!/bin/bash
pip install -q scikit-learn tqdm
python train_v2.py \
    --data_path $TRAIN_DATA_PATH \
    --epochs 30 \
    --batch_size 256 \
    --embedding_dim 32 \
    --hidden_dims 128 64 32 \
    --device cuda \
    --save_dir $TRAIN_CKPT_PATH
EOF

# 3. 上传到平台并运行
```

### 6.2 评估阶段

```bash
# 1. 准备推理脚本
# 已创建 deploy/infer.py

# 2. 创建prepare.sh
cat > deploy/prepare.sh << 'EOF'
#!/bin/bash
pip install -q scikit-learn tqdm
EOF

# 3. 上传到平台并评估
```

---

## 七、常见问题

### 7.1 训练超时
- 减少epoch数
- 增大batch_size
- 简化模型架构

### 7.2 推理超时（30分钟限制）
- 使用batch推理
- 减少模型复杂度
- 优化数据加载

### 7.3 Checkpoint不识别
- 确保目录名以`global_step`开头
- 检查字符限制
- 检查目录名长度（<300字符）

### 7.4 评估失败
- 检查infer.py是否包含main()函数
- 检查predictions.json格式
- 检查user_id是否匹配

---

## 八、快速检查清单

### 训练任务
- [ ] run.sh存在且可执行
- [ ] 数据路径使用$TRAIN_DATA_PATH
- [ ] 模型保存到$TRAIN_CKPT_PATH
- [ ] checkpoint目录以global_step开头
- [ ] TensorBoard写入$TRAIN_TF_EVENTS_PATH

### 评估任务
- [ ] infer.py存在且包含main()函数
- [ ] 输出predictions.json到$EVAL_RESULT_PATH
- [ ] predictions格式正确（user_id -> float）
- [ ] prepare.sh安装必要依赖
- [ ] 脚本总大小 < 100MB

---

*最后更新: 2026年5月19日*
