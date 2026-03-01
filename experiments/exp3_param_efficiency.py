from __future__ import annotations

import argparse
import json

import pandas as pd

from experiments.common import model_builders, out_paths
from targets.factory import build_target
from utils.config import DEFAULT_BUDGETS, DEFAULT_DIMS_MVP, DEFAULT_SEEDS, DEFAULT_TARGETS, DEFAULT_THRESHOLDS
from utils.param_matching import count_parameters, find_width_for_budget
from utils.seed import set_seed
from utils.training import TrainConfig, make_dataset, train_model


def parse_args():
    p = argparse.ArgumentParser(description="Experiment 3: parameter efficiency")
    p.add_argument("--dims", nargs="+", type=int, default=DEFAULT_DIMS_MVP)
    p.add_argument("--targets", nargs="+", type=str, default=DEFAULT_TARGETS)
    p.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    p.add_argument("--budgets", nargs="+", type=int, default=DEFAULT_BUDGETS)
    p.add_argument("--thresholds", nargs="+", type=float, default=DEFAULT_THRESHOLDS)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--results_root", type=str, default="results")
    p.add_argument("--n_train", type=int, default=5000)
    p.add_argument("--n_val", type=int, default=1000)
    p.add_argument("--n_test", type=int, default=1000)
    p.add_argument("--epochs", type=int, default=200)
    return p.parse_args()


def run_one(args):
    paths = out_paths(args.results_root)
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)

    builders = model_builders()
    rows = []
    for seed in args.seeds:
        set_seed(seed)
        for d in args.dims:
            for target_name in args.targets:
                target = build_target(target_name, d=d, seed=seed)
                dataset = make_dataset(target, d, args.n_train, args.n_val, args.n_test, seed, args.device)
                for pt in [True, False]:
                    for model_name, builder in builders.items():
                        tested = []
                        for budget in sorted(args.budgets):
                            match = find_width_for_budget(builder, d, args.depth, budget, passthrough=pt)
                            model = builder(input_dim=d, hidden_dim=match["width"], depth=args.depth, passthrough=pt)
                            metrics = train_model(model, dataset, TrainConfig(epochs=args.epochs), args.device)
                            tested.append((count_parameters(model), metrics["test_mse"], metrics["test_rel_err"]))
                        for thr in args.thresholds:
                            valid = [t for t in tested if t[1] <= thr]
                            min_params = min(v[0] for v in valid) if valid else float("inf")
                            row = {
                                "exp": "exp3",
                                "seed": seed,
                                "d": d,
                                "target": target_name,
                                "model": model_name,
                                "passthrough": pt,
                                "threshold": thr,
                                "min_param_for_mse": min_params,
                            }
                            print(json.dumps(row, ensure_ascii=False))
                            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(paths["raw"] / "exp3_param_efficiency.csv", index=False)
    agg = df.groupby(["d", "target", "model", "passthrough", "threshold"], as_index=False).agg(
        min_param_mean=("min_param_for_mse", "mean"),
        min_param_std=("min_param_for_mse", "std"),
    )
    agg.to_csv(paths["processed"] / "exp3_param_efficiency_agg.csv", index=False)


if __name__ == "__main__":
    run_one(parse_args())
