"""Demonstration script for uobench."""

from pathlib import Path

from uobench.cli import cmd_generate, cmd_report
from uobench.core.witness import verify
from uobench.io import load_instance
from uobench.solvers.alm import solve_alm
from uobench.solvers.gd import solve_gd
from uobench.solvers.prox import solve_fista


def main() -> None:
    out = Path("./datasets")
    args_gen = type(
        "Args",
        (),
        {
            "suite": "core18",
            "scales": "S",
            "problems": "A1_QP,A4_ECQP,B1_LASSO,D2_BP,C2_LCP",
            "out": str(out),
            "seed": 42,
            "extreme": False,
        },
    )
    cmd_generate(args_gen)

    a4_dir = out / "core18_S" / "A4_ECQP"
    inst = sorted(a4_dir.iterdir())[0]
    meta, arrays = load_instance(inst)
    assert verify(meta["id"], meta, arrays)
    res_alm = solve_alm(meta["id"], arrays, max_iter=10)
    print("ALM final KKT", res_alm["history"]["kkt"][-1] if res_alm["history"]["kkt"] else None)

    a1_dir = out / "core18_S" / "A1_QP"
    inst = sorted(a1_dir.iterdir())[0]
    meta, arrays = load_instance(inst)
    res_gd = solve_gd(meta["id"], arrays, max_iter=20)
    print("GD final obj", res_gd["obj"])

    b1_dir = out / "core18_S" / "B1_LASSO"
    inst = sorted(b1_dir.iterdir())[0]
    meta, arrays = load_instance(inst)
    res_fista = solve_fista(meta["id"], arrays, max_iter=20)
    print("FISTA obj", res_fista["obj"])

    args_report = type(
        "Args",
        (),
        {
            "root": str(out / "core18_S"),
            "save_md": "./reports/summary.md",
            "save_csv": "./reports/summary.csv",
            "save_json": "./reports/summary.json",
        },
    )
    cmd_report(args_report)


if __name__ == "__main__":
    main()
