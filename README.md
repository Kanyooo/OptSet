# UOBGen: Unified Optimization Benchmark Generator

UOBGen 提供 18 类经典优化问题的数据集生成器。研究者可以使用 Python API 或命令行快速构建具有可调参数、可重复的仿真数据集，用于评测优化算法。

## 安装

```bash
pip install -e .
```

## 快速开始

列出可用问题及套件：

```bash
uobgen list
```

生成默认核心套件（核心 18 个问题，S/M/L 规模）：

```bash
uobgen generate --suite core18 --out ./data --seed 42
```

只生成部分问题与规模：

```bash
uobgen generate --problems A1_QP,B1_LASSO --scales S --out ./data_s --seed 7
```

验证生成数据（结构性、条件数、SNR 等）：

```bash
uobgen verify --path ./data
```

## 项目结构

- `uobgen/`：Python 包，包含问题生成器与工具函数。
- `config/suites/`：YAML 套件定义。
- `DATASET_GUIDE.md`：中文用户指南与数学规格。
- `tests/`：pytest 单元测试。

## 许可

本项目基于 Apache-2.0 许可发布。

---

## ICNN论文实验MVP（Experiment 2/3/4）

本仓库新增了一个独立实验子系统，用于论文中的：
- 4.2 Expressivity under matched parameter budgets（实验2）
- 4.3 Parameter efficiency to reach target accuracy（实验3）
- 4.4 Ablation on passthrough and module design + 4.5 Training dynamics and geometric visualization（实验4）

### 1) 依赖

实验代码使用 PyTorch / pandas / matplotlib：

```bash
pip install torch pandas matplotlib
```

### 2) 模型定义与约束

已实现模型：
- ReLU-ICNN
- Softplus-ICNN
- Quad-ICNN
- Norm-ICNN
- SOC-ICNN

ELU-ICNN 默认**自动跳过**，原因：ELU 在全实数域上不满足全局凸性，违背此处 ICNN 凸性设定。

所有非负参数都通过 `softplus(raw_param)` 映射实现（不使用 hard clamp），覆盖：
- `U_l`, `c`
- `alpha_h`, `lambda_g`

`passthrough` 统一定义：
- `with passthrough`：保留 `v^T x + b_0`
- `without passthrough`：令该直连仿射项为 0（隐藏层与 Quad/Norm 模块保留）

### 3) 目标函数族（MVP）

MVP 默认覆盖六类：
- `quadratic`
- `norm_cone`
- `quadratic_plus_norm`
- `huber`
- `logsumexp`
- `structured_composition`

MVP 默认维度：`d in {2, 20, 50}`，且代码结构兼容后续扩展到 `{100, 200}`。

### 4) 运行方式

分别运行：

```bash
python experiments/exp2_matched_budget.py
python experiments/exp3_param_efficiency.py
python experiments/exp4_ablation.py
python experiments/summarize_results.py
```

一键运行：

```bash
python experiments/run_all.py
```

### 5) 多种子支持

所有主脚本均支持 `--seeds`（默认 `[0]`）。当前默认只跑单 seed，但聚合逻辑已兼容多 seed：
- 单 seed：输出单值
- 多 seed：输出 mean ± std

### 6) 输出目录

结果会写入：
- `results/raw/`：原始逐运行记录与 loss curves JSON
- `results/processed/`：聚合 CSV
- `results/figures/`：预算曲线、loss 曲线、2D 几何可视化
- `results/tables/`：论文整理用表格

### 7) 论文风格实验说明（精炼版）

- **实验2（同参数预算表达能力）**：在 matched parameter budgets 下比较不同模型 test MSE/relative error，检验 SOC-ICNN 是否更强表达。
- **实验3（参数效率）**：给定误差阈值，统计达到阈值所需最小参数量，检验 SOC-ICNN 是否更省参数。
- **实验4（消融+动态+几何）**：比较 passthrough 有无、模块组成（ReLU/Quad/Norm/SOC），并提供训练曲线和 d=2 的函数几何拟合可视化。
