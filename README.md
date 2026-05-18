# 2026腾讯广告算法大赛

**比赛官网**: https://algo.qq.com/

**GitHub仓库**: https://github.com/FrozenYears/TencentAdAlgorithmComp

---

## 快速开始

### 1. 克隆仓库
```bash
git clone https://github.com/FrozenYears/TencentAdAlgorithmComp.git
cd TencentAdAlgorithmComp
```

### 2. 创建虚拟环境
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/Mac
python -m venv .venv
source .venv/bin/activate
```

### 3. 安装依赖
```bash
pip install -r requirements.txt
```

### 4. 下载数据
```bash
python -c "from datasets import load_dataset; ds = load_dataset('TAAC2026/data_sample_1000'); ds['train'].to_parquet('data/taac2026_demo.parquet')"
```

### 5. 运行训练
```bash
cd src
python train_esmm.py --epochs 10 --device cpu
```

---

## 项目结构

```
TencentAdAlgorithmComp2026/
├── src/                          # 源代码
│   ├── feature_engineering.py    # 特征工程（120列处理）
│   ├── esmm_din_model.py         # ESMM+DIN模型
│   ├── evaluate.py               # 评估工具
│   ├── train_esmm.py             # 训练脚本（推荐）
│   ├── lgbm_cvr.py               # LightGBM方案
│   ├── ensemble_cvr.py           # 集成方案（比赛禁止）
│   └── baseline_cvr.py           # 原始baseline
├── research/                     # 研究文档（9份）
├── docs/                         # 计划文档
├── work_logs/                    # 工作日志
├── data/                         # 数据文件（需下载）
├── requirements.txt              # 依赖列表
└── README.md                     # 本文件
```

---

## 模型方案

| 方案 | 文件 | CVR AUC | 说明 |
|------|------|---------|------|
| Baseline | baseline_cvr.py | 0.5641 | 有bug，已废弃 |
| LightGBM | lgbm_cvr.py | 0.6893 | 快速方案 |
| 集成方案 | ensemble_cvr.py | ≥0.7 | 比赛禁止使用 |
| **ESMM+DIN** | **train_esmm.py** | **0.7528** | **推荐方案** |

---

## 比赛信息

- **赛题**: 面向大规模推荐的序列建模与特征交互的统一
- **任务**: 预测转化率（pCVR）
- **评估**: AUC-ROC
- **奖金**: 总奖池600万+人民币
- **截止**: 第一轮 2026年5月23日，第二轮 2026年6月24日

---

## 研究文档

详见 `research/` 目录，包含：
- 官网信息研究
- 往届经验研究
- 技术路线研究
- 竞赛策略研究
- 工程优化研究
- 综合分析报告
- 头脑风暴记录
- 优化计划

---

## 联系方式

如有问题，请通过GitHub Issues反馈。

---

*最后更新: 2026年5月18日*
