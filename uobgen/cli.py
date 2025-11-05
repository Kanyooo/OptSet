"""Command line interface for UOBGen."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

from .io import save_instance
from .problems import PROBLEM_NAMES, PROBLEM_REGISTRY
from .utils.yaml_loader import load_yaml
from .verification import summarise_reports, verify_tree


PARAM_ALIASES = {"lambda": "lam"}


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _suite_paths() -> Iterable[Path]:
    config_dir = _project_root() / "config" / "suites"
    return sorted(config_dir.glob("*.yaml"))


def cmd_list(_: argparse.Namespace) -> None:
    print("Available problems:")
    for pid in sorted(PROBLEM_REGISTRY):
        print(f"  {pid:10s} - {PROBLEM_NAMES.get(pid, '')}")
    print("\nAvailable suites:")
    for path in _suite_paths():
        data = load_yaml(str(path))
        print(f"  {path.stem}: {data.get('description', '')}")


def _expand_grid(params: Dict[str, object]) -> List[Dict[str, object]]:
    keys = list(params.keys())
    values = []
    for key in keys:
        val = params[key]
        if isinstance(val, list):
            values.append(val)
        else:
            values.append([val])
    combos = []
    for combination in itertools.product(*values):
        combos.append({keys[i]: combination[i] for i in range(len(keys))})
    return combos


def _normalize_params(params: Dict[str, object]) -> Dict[str, object]:
    normalized: Dict[str, object] = {}
    for key, value in params.items():
        target = PARAM_ALIASES.get(key, key)
        normalized[target] = value
    return normalized


def cmd_generate(args: argparse.Namespace) -> None:
    base_seed = args.seed
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.suite:
        suite_path = _project_root() / "config" / "suites" / f"{args.suite}.yaml"
        suite_cfg = load_yaml(str(suite_path))
        problem_cfg = suite_cfg.get("problems", {})
    else:
        problem_cfg = {}
    selected = args.problems.split(",") if args.problems else sorted(PROBLEM_REGISTRY.keys())
    scales = args.scales.split(",") if args.scales else ["S"]
    counter = 0
    for pid in selected:
        if pid not in PROBLEM_REGISTRY:
            print(f"[warn] unknown problem {pid}")
            continue
        generator = PROBLEM_REGISTRY[pid]
        cfg = problem_cfg.get(pid, {})
        scale_cfg = cfg.get("scales", {})
        seeds = int(cfg.get("seeds", 1))
        for scale in scales:
            params = scale_cfg.get(scale, {})
            grids = _expand_grid(params)
            if not grids:
                grids = [{}]
            for grid in grids:
                for s_idx in range(seeds):
                    counter += 1
                    seed = base_seed + counter + s_idx
                    kwargs = _normalize_params(grid)
                    instance = generator(seed=seed, **kwargs)
                    instance_dir = save_instance(out_dir, instance, name=f"{pid}_{scale}_{counter}")
                    meta_path = instance_dir / "meta.json"
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    meta["meta"]["scale"] = scale
                    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
                    print(f"generated {instance['id']} ({scale}) -> {instance_dir}")


def cmd_verify(args: argparse.Namespace) -> None:
    root = Path(args.path)
    if not root.exists():
        print("Path does not exist")
        return
    reports = verify_tree(root)
    print(summarise_reports(reports))
    prop_failures = sum(not rep.properties.ok for rep in reports)
    solver_failures = sum(rep.solver.attempted and not rep.solver.ok for rep in reports)
    print()
    print(f"Property checks failed: {prop_failures}")
    print(f"Solver checks failed: {solver_failures}")


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="uobgen", description="Unified Optimization Benchmark generator")
    sub = parser.add_subparsers(dest="command")

    list_parser = sub.add_parser("list", help="List problems and suites")
    list_parser.set_defaults(func=cmd_list)

    gen_parser = sub.add_parser("generate", help="Generate dataset instances")
    gen_parser.add_argument("--suite", type=str, default=None, help="Suite identifier")
    gen_parser.add_argument("--problems", type=str, default=None, help="Comma separated problem IDs")
    gen_parser.add_argument("--scales", type=str, default="S", help="Comma separated scales (S,M,L)")
    gen_parser.add_argument("--out", type=str, required=True, help="Output directory")
    gen_parser.add_argument("--seed", type=int, default=0, help="Base random seed")
    gen_parser.set_defaults(func=cmd_generate)

    ver_parser = sub.add_parser("verify", help="Verify generated datasets")
    ver_parser.add_argument("--path", type=str, required=True, help="Path to generated data root")
    ver_parser.set_defaults(func=cmd_verify)

    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
