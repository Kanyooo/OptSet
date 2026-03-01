from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results_root", type=str, default="results")
    return p.parse_args()


def fmt(mean, std):
    if pd.isna(std):
        return f"{mean:.4e}"
    return f"{mean:.4e} ± {std:.2e}"


def main(args):
    root = Path(args.results_root)
    out_tables = root / "tables"
    out_tables.mkdir(parents=True, exist_ok=True)

    exp2 = pd.read_csv(root / "processed" / "exp2_matched_budget_agg.csv")
    exp2["test_mse"] = [fmt(m, s) for m, s in zip(exp2.test_mse_mean, exp2.test_mse_std)]
    exp2["rel_err"] = [fmt(m, s) for m, s in zip(exp2.rel_err_mean, exp2.rel_err_std)]
    exp2.to_csv(out_tables / "exp2_main_table.csv", index=False)

    exp3 = pd.read_csv(root / "processed" / "exp3_param_efficiency_agg.csv")
    exp3["min_param"] = [fmt(m, s) for m, s in zip(exp3.min_param_mean, exp3.min_param_std)]
    exp3.to_csv(out_tables / "exp3_efficiency_table.csv", index=False)


if __name__ == "__main__":
    main(parse_args())
