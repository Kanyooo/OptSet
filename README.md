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
