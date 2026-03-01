from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

from experiments.common import model_builders, out_paths
from targets.factory import build_target
from utils.param_matching import count_parameters
from utils.seed import set_seed
from utils.training import TrainConfig, dump_curve, make_dataset, train_model
from utils.plotting import plot_loss_curves


def parse_args():
    p = argparse.ArgumentParser(description="Experiment 4: ablation + dynamics + 2D geometry")
    p.add_argument("--d", type=int, default=2)
    p.add_argument("--target", type=str, default="structured_composition")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--results_root", type=str, default="results")
    p.add_argument("--n_train", type=int, default=5000)
    p.add_argument("--n_val", type=int, default=1000)
    p.add_argument("--n_test", type=int, default=1000)
    p.add_argument("--epochs", type=int, default=200)
    return p.parse_args()


def surface_plot(fn, out_file: Path, title: str):
    grid = torch.linspace(-1, 1, 120)
    xx, yy = torch.meshgrid(grid, grid, indexing="ij")
    pts = torch.stack([xx.flatten(), yy.flatten()], dim=-1)
    with torch.no_grad():
        zz = fn(pts).reshape(xx.shape).cpu().numpy()
    plt.figure(figsize=(5, 4))
    cs = plt.contourf(xx.numpy(), yy.numpy(), zz, levels=40)
    plt.colorbar(cs)
    plt.title(title)
    plt.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=180)
    plt.close()


def run_one(args):
    paths = out_paths(args.results_root)
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    builders = model_builders()
    target = build_target(args.target, d=args.d, seed=args.seed)
    dataset = make_dataset(target, args.d, args.n_train, args.n_val, args.n_test, args.seed, args.device)

    rows = []
    curves = {}
    for pt in [True, False]:
        for model_name in ["relu_icnn", "quad_icnn", "norm_icnn", "soc_icnn", "softplus_icnn"]:
            model = builders[model_name](input_dim=args.d, hidden_dim=args.width, depth=args.depth, passthrough=pt)
            metrics = train_model(model, dataset, TrainConfig(epochs=args.epochs), args.device)
            key = f"{model_name}_pt{int(pt)}"
            curves[key] = metrics["curve"]
            dump_curve(metrics["curve"], paths["raw"] / "curves" / f"exp4_{key}_{args.target}.json")
            rows.append(
                {
                    "exp": "exp4",
                    "seed": args.seed,
                    "d": args.d,
                    "target": args.target,
                    "model": model_name,
                    "passthrough": pt,
                    "param_count": count_parameters(model),
                    **{k: v for k, v in metrics.items() if k != "curve"},
                }
            )
            if args.d == 2:
                surface_plot(lambda x: model(x.to(args.device)).cpu(), paths["figures"] / f"exp4_surface_{key}.png", key)

    if args.d == 2:
        surface_plot(target.evaluate, paths["figures"] / "exp4_surface_target.png", f"target-{args.target}")

    plot_loss_curves(curves, paths["figures"] / f"exp4_loss_curves_{args.target}.png")
    df = pd.DataFrame(rows)
    df.to_csv(paths["tables"] / "exp4_ablation_table.csv", index=False)
    print(json.dumps({"saved": str(paths['tables'] / 'exp4_ablation_table.csv')}))


if __name__ == "__main__":
    run_one(parse_args())
