"""Python-only workflow demo for uobench."""

from __future__ import annotations

from pathlib import Path

from uobench.core import diagnostics, report, witness
from uobench.core.spec import PROBLEM_REGISTRY
from uobench.io import load_instance, save_instance
from uobench.solvers.alm import solve_alm
from uobench.solvers.gd import solve_gd
from uobench.solvers.prox import solve_fista


def prepare_instances(root: Path) -> list[Path]:
    """Generate a handful of S-scale instances directly via the Python API."""
    root.mkdir(parents=True, exist_ok=True)
    selections = {
        "A1_QP": {"n": 200, "kappa": 1e3},
        "A4_ECQP": {"n": 200, "p": 20, "kappa": 1e2},
        "B1_LASSO": {"m": 200, "n": 400, "sparsity": 0.1},
        "D2_BP": {"m": 120, "n": 240, "s": 10},
        "C2_LCP": {"n": 60, "delta": 1e-2},
    }
    saved_paths: list[Path] = []
    for pid, knobs in selections.items():
        spec = PROBLEM_REGISTRY[pid]
        instance = spec.generator(seed=2024, knobs=knobs, extreme=False)
        arrays = instance["data"]
        meta = {
            "id": spec.problem_id,
            "name": spec.name,
            "family": spec.family,
            "seed": 2024,
            "dims": instance["dims"],
            "knobs": instance["knobs"],
            "witness": instance.get("witness", {}),
            "reference": {"has_reference": bool(instance.get("reference"))},
            "diagnostics": diagnostics.compute(spec.problem_id, arrays),
        }
        seed_tag = f"seed_{meta['seed']:04d}"
        paths = save_instance(root, "python_demo", pid, "S", seed_tag, meta, arrays, instance.get("readme", spec.name))
        saved_paths.append(paths.meta.parent)
    return saved_paths


def solve_subset(instance_dirs: list[Path]) -> None:
    """Run baseline solvers with matplotlib plots enabled."""
    for path in instance_dirs:
        meta, arrays = load_instance(path)
        print(f"Loaded {meta['id']} from {path}")  # noqa: T201
        ok = witness.verify(meta["id"], meta, arrays)
        print("  witness verified:", ok)  # noqa: T201
        if meta["id"] == "A4_ECQP":
            res = solve_alm(meta["id"], arrays, max_iter=20, plot=True)
        elif meta["id"] == "B1_LASSO":
            res = solve_fista(meta["id"], arrays, max_iter=50, plot=True)
        else:
            res = solve_gd(meta["id"], arrays, max_iter=50, plot=True)
        print("  final status:", res["status"], "objective:", res["obj"])  # noqa: T201


def emit_report(instance_dirs: list[Path], dest: Path) -> None:
    """Aggregate feasibility diagnostics into Markdown/CSV/JSON reports."""
    dest.mkdir(parents=True, exist_ok=True)
    rows = report.summarize_instances(instance_dirs)
    report.write_markdown(dest / "summary.md", rows)
    report.write_csv(dest / "summary.csv", rows)
    report.write_json(dest / "summary.json", rows)


def main() -> None:
    datasets_root = Path("./datasets_python")
    instance_dirs = prepare_instances(datasets_root)
    solve_subset(instance_dirs)
    emit_report(instance_dirs, Path("./reports_python"))


if __name__ == "__main__":
    main()
