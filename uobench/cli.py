"""Command line interface for uobench."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from .core.spec import PROBLEM_REGISTRY, SUITE_SPECS
from .core import diagnostics, report, witness
from .io import load_instance, save_instance
from .utils.rng import RNG
from .solvers import bb, gd, newton, alm, prox

SOLVER_DISPATCH = {
    "gd": gd.solve_gd,
    "bb": bb.solve_bb,
    "newton": newton.solve_newton,
    "alm": alm.solve_alm,
    "fista": prox.solve_fista,
    "pg": prox.solve_projected_gd,
}


def cmd_list(args: argparse.Namespace) -> None:
    print("Problems:")
    for spec in PROBLEM_REGISTRY.values():
        print(f"  {spec.problem_id}: {spec.name} [{spec.family}]")
    print("\nSuites:")
    for name, suite in SUITE_SPECS.items():
        print(f"  {name}: {suite.get('description', '')}")


def expand_suite(suite: Dict, scales: List[str], problems: List[str]) -> List[Dict]:
    worklist: List[Dict] = []
    for pid, pdata in suite["problems"].items():
        if problems and pid not in problems:
            continue
        for scale in scales:
            if scale not in pdata:
                continue
            entry = pdata[scale]
            for knob in entry.get("knobs", []):
                for seed in entry.get("seeds", [0]):
                    worklist.append({
                        "problem_id": pid,
                        "scale": scale,
                        "seed": int(seed),
                        "knobs": knob,
                    })
    return worklist


def cmd_generate(args: argparse.Namespace) -> None:
    suite_name = args.suite
    if suite_name not in SUITE_SPECS:
        raise SystemExit(f"Unknown suite {suite_name}")
    suite = SUITE_SPECS[suite_name]
    scales = args.scales.split(",") if args.scales else ["S"]
    problems = args.problems.split(",") if args.problems else []
    tasks = expand_suite(suite, scales, problems)
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    base_rng = RNG(args.seed)
    for idx, task in enumerate(tasks):
        spec = PROBLEM_REGISTRY[task["problem_id"]]
        rng_seed = base_rng.spawn(idx + task["seed"]).seed
        instance = spec.generator(rng_seed, task["knobs"], args.extreme)
        arrays = instance["data"]
        meta = {
            "id": spec.problem_id,
            "name": spec.name,
            "family": spec.family,
            "seed": rng_seed,
            "dims": instance["dims"],
            "knobs": instance.get("knobs", task["knobs"]),
            "witness": instance.get("witness", {}),
            "reference": {"has_reference": bool(instance.get("reference"))},
        }
        diag = diagnostics.compute(spec.problem_id, arrays)
        meta["diagnostics"] = diag
        readme = instance.get("readme", f"Instance for {spec.problem_id} with knobs {task['knobs']}")
        seed_tag = f"seed_{rng_seed:04d}"
        save_instance(root, suite_name, spec.problem_id, task["scale"], seed_tag, meta, arrays, readme)
        print(f"Saved {spec.problem_id} ({task['scale']}) at seed {rng_seed}")


def cmd_report(args: argparse.Namespace) -> None:
    root = Path(args.root)
    paths = sorted(p.parent for p in root.rglob("meta.json"))
    rows = report.summarize_instances(paths)
    if args.save_md:
        report.write_markdown(Path(args.save_md), rows)
    if args.save_csv:
        report.write_csv(Path(args.save_csv), rows)
    if args.save_json:
        report.write_json(Path(args.save_json), rows)
    total = len(rows)
    feasible = sum(1 for r in rows if r.get("feasible") == "yes")
    print(f"Collected {total} instances; {feasible} passed witness verification.")
    families = sorted({r.get("family", "") for r in rows})
    print("Families covered:", ", ".join(families))
    for row in rows:
        print(json.dumps(row, indent=2))


def cmd_solve(args: argparse.Namespace) -> None:
    inst_dir = Path(args.path)
    if args.solver not in SOLVER_DISPATCH:
        raise SystemExit(f"Unknown solver {args.solver}")
    meta, arrays = load_instance(inst_dir)
    solver_fn = SOLVER_DISPATCH[args.solver]
    result = solver_fn(
        problem_id=meta["id"],
        arrays=arrays,
        max_iter=args.max_iter,
        tol=args.tol,
        plot=args.plot,
    )
    print(json.dumps({"status": result["status"], "iters": result["iters"], "final": result["history"].get("f", [])[-1] if result["history"].get("f") else None}, indent=2))


def cmd_load_info(args: argparse.Namespace) -> None:
    meta, arrays = load_instance(Path(args.path))
    print(json.dumps(meta, indent=2))
    print("Arrays:", list(arrays.keys()))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="uobench")
    sub = parser.add_subparsers(dest="cmd")

    p_list = sub.add_parser("list")
    p_list.set_defaults(func=cmd_list)

    p_gen = sub.add_parser("generate")
    p_gen.add_argument("--suite", required=True)
    p_gen.add_argument("--scales", default="S")
    p_gen.add_argument("--problems", default="")
    p_gen.add_argument("--out", default="./datasets")
    p_gen.add_argument("--seed", type=int, default=0)
    p_gen.add_argument("--extreme", action="store_true")
    p_gen.set_defaults(func=cmd_generate)

    p_report = sub.add_parser("report")
    p_report.add_argument("--root", required=True)
    p_report.add_argument("--save-md", default="")
    p_report.add_argument("--save-csv", default="")
    p_report.add_argument("--save-json", default="")
    p_report.set_defaults(func=cmd_report)

    p_solve = sub.add_parser("solve")
    p_solve.add_argument("--path", required=True)
    p_solve.add_argument("--solver", required=True)
    p_solve.add_argument("--max-iter", type=int, default=500)
    p_solve.add_argument("--tol", type=float, default=1e-6)
    p_solve.add_argument("--plot", action="store_true", help="Display descent curve via matplotlib")
    p_solve.set_defaults(func=cmd_solve)

    p_info = sub.add_parser("load-info")
    p_info.add_argument("--path", required=True)
    p_info.set_defaults(func=cmd_load_info)

    return parser


def main(argv: List[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
