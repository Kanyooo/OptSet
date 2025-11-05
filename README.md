# uobench

Unified Optimization Benchmark providing certified-feasible synthetic datasets, diagnostics, and baseline solvers for small-scale sanity checks.

## Features
- 18 generator families spanning smooth, nonsmooth, variational inequality, and conic problems.
- Difficulty knobs with optional `--extreme` resampling to realize pathological behavior.
- Witness certificates and structural diagnostics saved with every instance.
- Baseline first-order and second-order solvers (GD, BB, Newton, ALM, ISTA/FISTA, projected methods).
- Command line interface for listing problems, generating suites, reporting feasibility, and running solvers.

## Quick Start (PyCharm project style)
1. 克隆仓库后直接在 PyCharm 中以普通项目方式打开，无需执行 `pip install -e .`。
2. 在项目解释器中安装依赖包：`numpy`、`scipy`、`matplotlib`、`pytest`（仅在需要运行测试时）。
3. 运行根目录下的 `example.py`：
   ```bash
   python example.py
   ```
   该脚本展示如何在 Python 中生成数据集、调用求解器（会弹出 Matplotlib 下降曲线）以及导出报告。

如需使用命令行工具（可选），可直接通过模块方式调用：
```bash
python -m uobench.cli list
python -m uobench.cli generate --suite core18 --scales S --out ./datasets --seed 42
python -m uobench.cli report --root ./datasets/core18_S --save-md ./reports/summary.md --save-csv ./reports/summary.csv
python -m uobench.cli solve --path ./datasets/core18_S/A1_QP/seed_0042 --solver gd --max-iter 200 --plot
```

## Python API Usage

```python
from pathlib import Path

from uobench.io import load_instance
from uobench.solvers.gd import solve_gd
from uobench.core.witness import verify

# Load a generated instance
meta, arrays = load_instance(Path("./datasets/core18_S/A4_ECQP/seed_0042"))

# Check the stored feasibility certificate
assert verify(meta["id"], meta, arrays)

# Run a baseline solver并绘制下降曲线
result = solve_gd(meta["id"], arrays, max_iter=200, tol=1e-6, plot=True)
print("status:", result["status"], "iterations:", result["iters"])  # noqa: T201
```

## Repository Layout
```
uobench/
  cli.py
  core/
  problems/
  solvers/
  utils/
config/suites/core18.yaml
example.py
```

## License
Apache-2.0. See [LICENSE](LICENSE).
