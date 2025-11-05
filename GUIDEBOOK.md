# uobench 使用指南

本指南面向希望在 `uobench` 基准库上开展研究或扩展开发的用户，涵盖以下主题：

1. 如何安装与快速上手（CLI 与 Python API）。
2. 如何基于现有生成器自定义参数与批量生成数据集。
3. 如何向基准库新增自定义问题生成器/数据集。
4. 如何实现并接入新的求解器以运行小规模可行性验证。

> **提示**：全部示例均假设当前工作目录为仓库根目录，且已通过 `pip install -e .` 完成本地开发安装。

---

## 1. 安装与快速上手

### 1.1 安装
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

安装完成后，可通过 `uobench list` 查看基准库中注册的问题与套件：
```bash
uobench list
```

### 1.2 生成实例
使用 CLI 的 `generate` 命令即可批量生成数据集。例如，只生成 `core18` 套件的 S 规模并强制 "极端" 难度：
```bash
uobench generate \
  --suite core18 \
  --scales S \
  --out ./datasets \
  --seed 42 \
  --extreme true
```

生成后的目录结构如下：
```
datasets/
  core18_S/
    A1_QP/
      seed_42_0001/
        meta.json
        data.npz
        README.md
    ...
```

### 1.3 查看与验证
- `uobench load-info --path <instance_dir>`：打印某个实例的元信息、证书和诊断指标。
- `uobench report --root <suite_dir> --save-md report.md --save-csv summary.csv`：
  - 自动遍历目录，验证可行性证书；
  - 汇总诊断指标、极端性；
  - 输出 Markdown/CSV 报告，便于归档。

### 1.4 运行基线求解器
示例：对 `A1_QP` 实例运行梯度下降（GD）：
```bash
uobench solve \
  --path datasets/core18_S/A1_QP/seed_42_0001 \
  --solver gd \
  --max-iter 2000 \
  --tol 1e-8
```

更复杂的等式约束问题可以调用增广拉格朗日法（ALM）：
```bash
uobench solve \
  --path datasets/core18_S/A4_ECQP/seed_42_0001 \
  --solver alm \
  --inner newton \
  --max-iter 200 \
  --tol 1e-6
```

在 Python 中直接调用示例，可参考仓库中的 [`example.py`](example.py)。

---

## 2. 自定义参数与批量生成

### 2.1 修改套件配置
套件配置存储于 [`config/suites/core18.yaml`](config/suites/core18.yaml)。文件中每个问题都指定了 `S/M/L` 规模、网格参数以及随机种子数量。若想调整参数或新增自定义尺度：
1. 复制 `core18.yaml` 为新的配置文件，例如 `config/suites/my_suite.yaml`。
2. 编辑所需问题的 `knobs`、`scales`、`seeds`。所有可用旋钮均可在 [`uobench/core/spec.py`](uobench/core/spec.py) 中的注册表条目中找到，字段名称需与生成器实现一致。
3. 运行：
   ```bash
   uobench generate --suite my_suite --config config/suites/my_suite.yaml --out ./datasets --seed 123
   ```
   若未指定 `--config`，CLI 默认读取 `config/suites/<suite>.yaml`。

### 2.2 直接调用生成器
若只需一次性生成某个特定参数组合，可通过问题注册表直接调用：
```python
from uobench.core.spec import PROBLEM_REGISTRY

spec = PROBLEM_REGISTRY["A1_QP"]
instance = spec.generator(seed=2024, knobs={"n": 200, "kappa": 1e4}, extreme=False)
data = instance["data"]  # 包含 Q、b、x_star 等数组
```
返回的 `instance` 字典包含 `data`、`dims`、`knobs`、`witness`、`readme` 等字段，可配合 `uobench.io.save_instance` 落盘：
```python
from pathlib import Path
from uobench.io import save_instance

root = Path("./custom")
meta = {
    "id": "A1_QP",
    "name": "Strongly-convex QP",
    "family": "smooth",
    "seed": 2024,
    "dims": instance["dims"],
    "knobs": instance["knobs"],
    "witness": instance["witness"],
    "diagnostics": {},
}
save_instance(root=root, suite="play", problem_id="A1_QP", scale="S", seed_tag="seed_2024", meta=meta, arrays=data, readme=instance["readme"])
```

### 2.3 控制极端难度
CLI/生成函数均接受 `extreme` 标志。当为 `True` 时，生成器会自动将旋钮推向更具挑战性的配置，例如提高条件数、缩小约束集或提升噪声强度；最终数值会记录在生成目录的 `meta.json` → `diagnostics` 字段中。若极端模式下仍无法满足需求，可手动设定更激进的旋钮，或在自定义生成器中加入额外筛选逻辑。

---

## 3. 新增自定义问题生成器

### 3.1 注册新问题
1. 在相应目录（例如 `uobench/problems/a_smooth.py`）中实现生成函数，遵循统一签名：
   ```python
   def generate_my_problem(seed: int, knobs: Dict, extreme: bool) -> Dict:
       rng = np.random.default_rng(seed)
       ...
       return {
           "data": arrays,
           "dims": {"n": n, ...},
           "knobs": knobs_used,
           "witness": witness,
           "reference": reference_dict,
           "readme": "...",
       }
   ```
2. 在 [`uobench/problems/__init__.py`](uobench/problems/__init__.py) 中导出该函数。
3. 在 [`uobench/core/spec.py`](uobench/core/spec.py) 的 `PROBLEM_REGISTRY` 中登记问题元数据（问题 ID、名称、家族分类等）。
4. 如需 CLI 支持，在 `config/suites/` 下的 YAML（JSON 格式）配置中添加对应条目，指定不同规模的旋钮网格与种子。

### 3.2 证书与诊断
- 证书逻辑：
  - 如问题有约束，务必构造原始/对偶可行点，并在返回字典的 `witness` 字段中保存。
  - 同时在 [`uobench/core/witness.py`](uobench/core/witness.py) 中实现校验逻辑或复用现有函数。
- 诊断指标：
  - 计算数值难度信息（条件数、最小特征值、SNR 等）。
  - 调用 [`uobench/core/diagnostics.py`](uobench/core/diagnostics.py) 中的辅助函数以保持格式一致。

### 3.3 测试
新增问题后，请在 `tests/test_generation.py` 中补充断言，确保维度、性质（SPD、可行性）等满足预期。

---

## 4. 编写自定义求解器

### 4.1 求解器接口
所有求解器位于 `uobench/solvers/`，统一返回结构：
```python
Result = Dict[str, Any]
{
    "x": ...,
    "obj": float | None,
    "history": {
        "f": [ ... ],
        "kkt": [ ... ],
        "prim_resid": [ ... ],
        ...
    },
    "status": "converged" | "stalled" | "max_iter",
    "iters": int,
}
```
如需新增自定义求解器：
1. 在 `uobench/solvers/` 下创建文件，例如 `my_solver.py`，实现函数 `solve_my_solver(problem_id: str, arrays: Dict[str, np.ndarray], **opts)`。
2. 在 [`uobench/solvers/__init__.py`](uobench/solvers/__init__.py) 中导出模块，并在 [`uobench/cli.py`](uobench/cli.py) 的 `SOLVER_DISPATCH` 字典中注册条目：
   ```python
   SOLVER_DISPATCH = {
       ...,
       "my-solver": solve_my_solver,
   }
   ```
3. 如需额外 CLI 选项，可在 `solve` 子命令解析器中添加参数并传递给求解器。
4. （可选）在 [`tests/test_solvers_smoke.py`](tests/test_solvers_smoke.py) 中添加烟雾测试，验证求解器在小规模实例上的运行结果。

### 4.2 利用已有工具
- 使用 [`uobench/solvers/kkt.py`](uobench/solvers/kkt.py) 中的辅助函数计算 KKT 残差。
- 对需要投影的约束，可复用 [`uobench/solvers/prox.py`](uobench/solvers/prox.py) 提供的投影与近端算子。
- 若求解器需要线性代数工具（例如 Haar 采样、Toeplitz 协方差、条件数估计），可直接使用 [`uobench/utils/linalg.py`](uobench/utils/linalg.py)。

### 4.3 与 CLI 集成
一旦在 `SOLVER_DISPATCH` 注册成功，CLI `uobench solve --solver my-solver` 即可调用。可通过 `--max-iter`、`--tol` 等通用参数控制收敛准则；若需要额外选项，可在 CLI 中新增专用参数并传递给求解器。

---

## 5. 常见问题（FAQ）

- **如何保证生成数据可行？** 每个实例都包含 `witness` 字段，可通过 `uobench report` 或 `uobench.core.witness.verify` 自动验证。
- **如何调试极端实例失败？** 查看 `meta.json` 中 `diagnostics` 字段是否已经接近阈值，必要时手动加大旋钮或关闭 `--extreme` 再重新生成。
- **如何扩展报告内容？** 可修改 [`uobench/core/report.py`](uobench/core/report.py) 中的 `write_markdown`/`write_csv`/`write_json` 函数，或在生成后自行解析 `rows` 列表生成自定义摘要。

---

通过上述步骤，您可以快速使用 `uobench` 提供的现成数据集与求解器，也可以根据研究需求灵活扩展问题类型、诊断指标和数值算法。祝研究顺利！
