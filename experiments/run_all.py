from __future__ import annotations

import argparse
import subprocess


def parse_args():
    p = argparse.ArgumentParser(description="Run MVP experiments 2/3/4")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--results_root", type=str, default="results")
    return p.parse_args()


def run(cmd):
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main(args):
    run(["python", "experiments/exp2_matched_budget.py", "--device", args.device, "--results_root", args.results_root])
    run(["python", "experiments/exp3_param_efficiency.py", "--device", args.device, "--results_root", args.results_root])
    run(["python", "experiments/exp4_ablation.py", "--device", args.device, "--results_root", args.results_root])
    run(["python", "experiments/summarize_results.py", "--results_root", args.results_root])


if __name__ == "__main__":
    main(parse_args())
