from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from experiments.common import model_builders, out_paths
from targets.factory import build_target
from utils.config import DEFAULT_BUDGETS, DEFAULT_DIMS_MVP, DEFAULT_SEEDS, DEFAULT_TARGETS
from utils.param_matching import count_parameters, find_width_for_budget
from utils.seed import set_seed
from utils.training import TrainConfig, dump_curve, make_dataset, train_model
from utils.plotting import plot_budget_curves


def parse_args():
    p = argparse.ArgumentParser(description="Experiment 2: matched parameter budget expressivity")
    p.add_argument("--dims", nargs="+", type=int, default=DEFAULT_DIMS_MVP)
    p.add_argument("--targets", nargs="+", type=str, default=DEFAULT_TARGETS)
    p.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    p.add_argument("--budgets", nargs="+", type=int, default=DEFAULT_BUDGETS)
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

    rows = []
    builders = model_builders()
    for seed in args.seeds:
        set_seed(seed)
        for d in args.dims:
            for target_name in args.targets:
                target = build_target(target_name, d=d, seed=seed)
                dataset = make_dataset(target, d, args.n_train, args.n_val, args.n_test, seed, args.device)
                for passthrough in [True, False]:
                    for budget in args.budgets:
                        for model_name, builder in builders.items():
                            match = find_width_for_budget(builder, d, args.depth, budget, passthrough=passthrough)
                            model = builder(input_dim=d, hidden_dim=match["width"], depth=args.depth, passthrough=passthrough)
                            metrics = train_model(model, dataset, TrainConfig(epochs=args.epochs), args.device)
                            n_params = count_parameters(model)
                            curve_name = f"exp2_seed{seed}_{target_name}_d{d}_{model_name}_pt{int(passthrough)}_b{budget}.json"
                            dump_curve(metrics["curve"], paths["raw"] / "curves" / curve_name)
                            row = {
                                "exp": "exp2",
                                "seed": seed,
                                "d": d,
                                "target": target_name,
                                "model": model_name,
                                "passthrough": passthrough,
                                "budget": budget,
                                "width": match["width"],
                                "match_rel_gap": match["rel_gap"],
                                "param_count": n_params,
                                **{k: v for k, v in metrics.items() if k != "curve"},
                            }
                            print(json.dumps(row, ensure_ascii=False))
                            rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = paths["raw"] / "exp2_matched_budget.csv"
    df.to_csv(csv_path, index=False)

    # aggregate single or multi-seed uniformly
    agg = df.groupby(["d", "target", "model", "passthrough", "budget"], as_index=False).agg(
        test_mse_mean=("test_mse", "mean"),
        test_mse_std=("test_mse", "std"),
        rel_err_mean=("test_rel_err", "mean"),
        rel_err_std=("test_rel_err", "std"),
        params_mean=("param_count", "mean"),
    )
    agg.to_csv(paths["processed"] / "exp2_matched_budget_agg.csv", index=False)

    for (d, target, pt), gdf in df.groupby(["d", "target", "passthrough"]):
        plot_budget_curves(gdf, paths["figures"] / f"exp2_budget_curves_d{d}_{target}_pt{int(pt)}.png")


if __name__ == "__main__":
    run_one(parse_args())
