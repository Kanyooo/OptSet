# uobench

Unified Optimization Benchmark providing certified-feasible synthetic datasets, diagnostics, and baseline solvers for small-scale sanity checks.

## Features
- 18 generator families spanning smooth, nonsmooth, variational inequality, and conic problems.
- Difficulty knobs with optional `--extreme` resampling to realize pathological behavior.
- Witness certificates and structural diagnostics saved with every instance.
- Baseline first-order and second-order solvers (GD, BB, Newton, ALM, ISTA/FISTA, projected methods).
- Command line interface for listing problems, generating suites, reporting feasibility, and running solvers.

## Quick Start
```bash
pip install -e .
uobench list
uobench generate --suite core18 --scales S --out ./datasets --seed 42
uobench report --root ./datasets/core18_S --save-md ./reports/summary.md --save-csv ./reports/summary.csv --save-json ./reports/summary.json
uobench solve --path ./datasets/core18_S/A1_QP/seed_0042 --solver gd --max-iter 200
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

# Run a baseline solver
result = solve_gd(meta["id"], arrays, max_iter=200, tol=1e-6)
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
