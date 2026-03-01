from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_budget_curves(df: pd.DataFrame, out_file: Path, value_col: str = "test_mse"):
    plt.figure(figsize=(7, 5))
    for model, gdf in df.groupby("model"):
        plt.plot(gdf["budget"], gdf[value_col], marker="o", label=model)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Parameter budget")
    plt.ylabel(value_col)
    plt.legend()
    plt.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=180)
    plt.close()


def plot_loss_curves(curves: dict, out_file: Path):
    plt.figure(figsize=(7, 5))
    for name, c in curves.items():
        plt.plot(c["train_mse"], label=f"{name}-train", alpha=0.85)
        plt.plot(c["val_mse"], label=f"{name}-val", linestyle="--", alpha=0.85)
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend(fontsize=8)
    plt.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=180)
    plt.close()
