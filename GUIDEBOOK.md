# uobench 实战指南（PyCharm 项目版）

本指南面向希望在 **PyCharm** 或任何本地 IDE 中，以普通 Python 项目的方式使用 `uobench` 的研究人员。内容覆盖：

1. 环境准备与项目结构概览。
2. 各核心模块/函数的作用、输入输出、返回格式说明。
3. 如何生成/扩展数据集与控制难度旋钮。
4. 如何在 Python 中调用基线求解器（含 Matplotlib 下降曲线）。
5. 如何新增自定义问题与自定义求解器。

> **重要提醒**：仓库无需打包安装。只要在 PyCharm 中将根目录作为项目打开，确保解释器安装 `numpy`、`scipy`、`matplotlib`、`pytest`（可选），即可运行所有脚本。

---

## 1. 环境准备

1. 克隆仓库，并在 PyCharm 中选择“Open”导入根目录。
2. 在 `File → Settings → Project → Python Interpreter` 中，为当前项目安装依赖：
   ```bash
   pip install numpy scipy matplotlib pytest
   ```
3. 直接运行根目录下的 `example.py` 可以验证环境是否就绪。该脚本会：
   - 通过 Python API 生成若干 S 规模实例；
   - 调用求解器并绘制下降曲线；
   - 导出 Markdown/CSV/JSON 报告。

项目目录结构：
```
uobench/
  cli.py                # 命令行入口（可选使用）
  core/                 # 规格、诊断、证书等核心逻辑
  problems/             # 18 个问题生成器
  solvers/              # GD/BB/Newton/ALM/FISTA 等求解器
  utils/                # RNG、线性代数、统计工具
config/suites/core18.yaml
example.py              # Python API 使用范例
```

---

## 2. 函数速查表

### 2.1 核心注册与规格

| 模块 | 函数 | 输入 | 输出 | 说明 |
| ---- | ---- | ---- | ---- | ---- |
| `uobench.core.spec` | `PROBLEM_REGISTRY` | 键为问题 ID 的字典 | `ProblemSpec` 对象 | 包含 `problem_id`、`name`、`family`、`generator` 等信息。可直接 `PROBLEM_REGISTRY["A1_QP"].generator(...)` 生成实例。 |
| `uobench.core.spec` | `SUITE_SPECS` | - | dict | 预定义套件（如 `core18`）的规模、参数网格与描述。CLI 与示例脚本均基于该字典展开任务。 |

### 2.2 I/O 与报告

| 模块 | 函数 | 输入 | 输出 | 说明 |
| ---- | ---- | ---- | ---- | ---- |
| `uobench.io` | `save_instance(root, suite, problem_id, scale, seed_tag, meta, arrays, readme)` | `root`: 根目录 `Path`；`suite`: 名称；`problem_id`: 问题 ID；`scale`: S/M/L；`seed_tag`: 目录名；`meta`: dict；`arrays`: dict；`readme`: str | `InstancePaths(meta, data, readme)` | 创建目录结构 `<root>/<suite>_<scale>/<problem_id>/<seed_tag>/`，写入 `meta.json`、`data.npz`、`README.md`。返回对象中包含三个文件路径。 |
| `uobench.io` | `load_instance(path)` | `path`: 指向单个实例的目录或 `Path` | `(meta: dict, arrays: dict[str, np.ndarray])` | 读取 `meta.json` 与 `data.npz`，方便后续验证或求解。 |
| `uobench.io` | `load_suite_index(root)` | `root`: 套件根目录 | dict | 返回 `path → meta` 的索引表，便于在 Python 中快速遍历全部实例。 |
| `uobench.core.report` | `summarize_instances(paths)` | `paths`: 多个实例目录的 `Iterable[Path]` | `List[Dict[str, str]]` | 为每个实例生成一行诊断摘要（含可行性、条件数、SNR 等）。 |
| `uobench.core.report` | `write_markdown(md_path, rows)` | Markdown 文件路径、`rows` 列表 | `None` | 将 `summarize_instances` 的结果写为 Markdown 表格和分组统计。 |
| `uobench.core.report` | `write_csv(csv_path, rows)` | CSV 文件路径 | `None` | 输出平面 CSV。 |
| `uobench.core.report` | `write_json(json_path, rows)` | JSON 文件路径 | `None` | 输出 JSON 列表格式，适合脚本解析。 |

### 2.3 诊断与证书

| 模块 | 函数 | 输入 | 输出 | 说明 |
| ---- | ---- | ---- | ---- | ---- |
| `uobench.core.diagnostics` | `compute(problem_id, arrays)` | `problem_id`: str；`arrays`: dict | dict | 返回条件数、最小特征值、SNR、投影角度等难度指标，字段随问题类型不同。 |
| `uobench.core.witness` | `verify(problem_id, meta, arrays, tol=1e-7)` | `problem_id`: str；`meta`: dict；`arrays`: dict；`tol`: float | bool | 检查 `meta["witness"]` 中的可行性证书是否满足约束（例如等式/不等式/互补条件）。报告命令与示例脚本均会调用该函数。

### 2.4 求解器（均支持 Matplotlib 绘图）

所有求解器返回统一格式的 `dict`：
```python
{
    "x": np.ndarray 或复合结构,
    "obj": float,
    "history": {
        "f": [ ... ],     # 迭代过程中的目标值/梯度范数等
        ...                 # 其他与算法相关的指标
    },
    "status": "converged" | "max_iter" | "stalled",
    "iters": int
}
```

| 模块 | 函数 | 关键输入 | 绘图开关 | 说明 |
| ---- | ---- | ---- | ---- | ---- |
| `uobench.solvers.gd` | `solve_gd(problem_id, arrays, max_iter=500, tol=1e-6, plot=False)` | `problem_id`：问题 ID；`arrays`：数据；`max_iter`、`tol`：终止条件 | `plot=True` 时会使用 Matplotlib 绘制目标值下降曲线。 | 对无约束或简单正则化问题的梯度下降（Armijo 回溯）。 |
| `uobench.solvers.bb` | `solve_bb(..., plot=False)` | 同上 | 同上 | Barzilai–Borwein 步长选择的梯度法。 |
| `uobench.solvers.newton` | `solve_newton(..., plot=False)` | 同上 | 同上 | 阻尼牛顿法，自动为非正定 Hessian 添加移位。 |
| `uobench.solvers.alm` | `solve_alm(problem_id, arrays, max_iter=100, tol=1e-6, inner='gd', plot=False)` | 需要 `arrays` 中包含 `A`、`d` 等约束信息 | 同上 | 增广拉格朗日法。`inner` 参数保留用于自定义内部解算器；当前默认按闭式/梯度法更新。 |
| `uobench.solvers.prox` | `solve_fista(problem_id, arrays, max_iter=100, tol=1e-6, plot=False)` | 适合 LASSO/Elastic Net/BP 等问题 | 同上 | ISTA/FISTA；`history['f']` 记录目标值。 |
| `uobench.solvers.prox` | `solve_projected_gd(problem_id, arrays, max_iter=200, tol=1e-6, plot=False)` | 支持盒约束、球约束、单纯形投影 | 同上 | 投影梯度法；`history['f']` 保存梯度范数，图像展示收敛过程。 |

> **绘图细节**：当 `plot=True` 时，函数内部会使用 `matplotlib` 的 `Agg` 后端绘制折线图并调用 `plt.show()`。在 PyCharm 中运行脚本会自动弹出窗口或在 SciView 中显示曲线。

---

## 3. 生成与管理数据集

### 3.1 直接使用 Python API

```python
from pathlib import Path
from uobench.core.spec import PROBLEM_REGISTRY
from uobench.core import diagnostics
from uobench.io import save_instance

spec = PROBLEM_REGISTRY["A1_QP"]
instance = spec.generator(seed=2024, knobs={"n": 200, "kappa": 1e3}, extreme=False)
arrays = instance["data"]
meta = {
    "id": spec.problem_id,
    "name": spec.name,
    "family": spec.family,
    "seed": 2024,
    "dims": instance["dims"],
    "knobs": instance["knobs"],
    "witness": instance["witness"],
    "diagnostics": diagnostics.compute(spec.problem_id, arrays),
    "reference": {"has_reference": bool(instance.get("reference"))},
}
paths = save_instance(Path("./datasets_python"), "demo", spec.problem_id, "S", "seed_2024", meta, arrays, instance["readme"])
print("保存路径:", paths.meta.parent)
```

### 3.2 控制极端/病态难度

所有生成器均接受 `extreme: bool` 参数：
```python
spec = PROBLEM_REGISTRY["A4_ECQP"]
inst = spec.generator(seed=2024, knobs={"n": 500, "p": 30, "kappa": 1e4}, extreme=True)
print(inst["diagnostics"]["cond_Q"])  # 极端模式下会显著增大条件数
```
如需更细粒度控制，可直接在 `knobs` 中设置想要的参数值（例如 `SNR`、`rho`、`delta` 等），并在外部循环中结合诊断指标实现自定义筛选。

### 3.3 批量生成（可选：命令行）

即使不安装为包，也可以使用模块形式调用 CLI：
```bash
python -m uobench.cli generate --suite core18 --scales S --out ./datasets --seed 7 --extreme
```
或者在 Python 中调用 `uobench.cli` 提供的函数（参考 `example.py`）。

---

## 4. 在 Python 中调用求解器

以下示例展示如何加载实例、验证证书并绘制下降曲线：

```python
from pathlib import Path
from uobench.io import load_instance
from uobench.core.witness import verify
from uobench.solvers.gd import solve_gd

inst_dir = Path("./datasets_python/demo/A1_QP/seed_2024")
meta, arrays = load_instance(inst_dir)
print("证书有效:", verify(meta["id"], meta, arrays))

result = solve_gd(meta["id"], arrays, max_iter=200, tol=1e-6, plot=True)
print("最终目标值:", result["obj"])
```

对于等式约束问题：
```python
from uobench.solvers.alm import solve_alm
result = solve_alm("A4_ECQP", arrays, max_iter=50, tol=1e-6, plot=True)
print("原始残差序列:", result["history"]["prim_resid"])
```

LASSO/Elastic Net：
```python
from uobench.solvers.prox import solve_fista
result = solve_fista("B1_LASSO", arrays, max_iter=200, plot=True)
```

所有求解器的 `history["f"]` 都会被绘制在图中，可用于检查下降质量、调试步长等。

---

## 5. 扩展：新增问题与自定义求解器

### 5.1 新增问题生成器

1. 在相应模块（如 `uobench/problems/a_smooth.py`）中实现函数：
   ```python
   def generate_my_problem(seed: int, knobs: dict, extreme: bool) -> dict:
       rng = np.random.default_rng(seed)
       # 构造问题数据
       arrays = {...}
       witness = {...}
       diagnostics = {...}
       return {
           "data": arrays,
           "dims": {"n": n, "m": m, "p": p},
           "knobs": knobs,
           "witness": witness,
           "reference": {"x_star": reference_solution},
           "diagnostics": diagnostics,
           "readme": "问题描述",
       }
   ```
2. 在 `uobench/problems/__init__.py` 中导出该函数。
3. 在 `uobench/core/spec.py` 的 `PROBLEM_REGISTRY` 中注册元信息。
4. 如需通过 CLI/批量配置使用，可在 `config/suites/*.yaml` 中加入对应条目。
5. 更新 `tests/test_generation.py` 以覆盖新问题的基本性质（维度、证书、条件数等）。

### 5.2 自定义求解器

1. 在 `uobench/solvers/` 下创建新的 Python 文件，实现函数：
   ```python
   def solve_my_solver(problem_id: str, arrays: Dict[str, np.ndarray], *, max_iter: int = 100, tol: float = 1e-6, plot: bool = False) -> Dict:
       history = {"f": []}
       # 迭代逻辑
       if plot:
           from uobench.solvers.gd import _maybe_plot
           _maybe_plot(history, f"MySolver on {problem_id}", "Metric", True)
       return {"x": x, "obj": objective, "status": status, "history": history, "iters": iters}
   ```
2. 在 `uobench/solvers/__init__.py` 中导入该模块；若需 CLI 支持，在 `uobench/cli.py` 的 `SOLVER_DISPATCH` 中注册。
3. 在 `tests/test_solvers_smoke.py` 中添加烟囱测试，确保 solver 能在 S 规模数据上收敛到合理残差。
4. 若求解器需要特殊图像或额外指标，可在返回的 `history` 中添加自定义字段，并在报告/绘图中解析。

---

## 6. 参考脚本：`example.py`

`example.py` 演示了完整的 Python 工作流：

1. `prepare_instances` 使用 `PROBLEM_REGISTRY` 和 `save_instance` 生成五个示例问题。
2. `solve_subset` 逐个载入实例、验证可行性证书，并调用 `solve_gd`/`solve_alm`/`solve_fista`，在每次求解时自动绘制下降曲线。
3. `emit_report` 调用 `uobench.core.report` 生成 Markdown、CSV、JSON 三种格式的汇总报告。

运行该脚本即可查看所有流程的标准写法，并可根据需要复制到自己的项目或 Notebook 中进一步扩展。

---

祝使用顺利！若在拓展过程中遇到特殊需求，可直接修改或新增模块，整个工程均以普通 Python 项目方式组织，无需打包发布。
