from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from utils.metrics import mse, relative_error


@dataclass
class TrainConfig:
    lr: float = 1e-3
    batch_size: int = 256
    epochs: int = 200
    patience: int = 25
    weight_decay: float = 0.0


def make_dataset(target_fn, d: int, n_train: int, n_val: int, n_test: int, seed: int, device: str):
    g = torch.Generator(device="cpu").manual_seed(seed)

    def sample(n: int):
        x = 2.0 * torch.rand((n, d), generator=g) - 1.0
        y = target_fn.evaluate(x)
        return x.to(device), y.to(device)

    return {"train": sample(n_train), "val": sample(n_val), "test": sample(n_test)}


def train_model(model: nn.Module, dataset: dict, cfg: TrainConfig, device: str):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.MSELoss()

    x_train, y_train = dataset["train"]
    x_val, y_val = dataset["val"]
    x_test, y_test = dataset["test"]

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=cfg.batch_size, shuffle=True)

    best = {"epoch": -1, "val": float("inf"), "state": None}
    logs = {"train_mse": [], "val_mse": []}
    start = time.time()
    stale = 0

    for epoch in range(cfg.epochs):
        model.train()
        for xb, yb in train_loader:
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            tr_pred = model(x_train)
            va_pred = model(x_val)
            tr = mse(tr_pred, y_train)
            va = mse(va_pred, y_val)
        logs["train_mse"].append(tr)
        logs["val_mse"].append(va)

        if va < best["val"]:
            best.update({"epoch": epoch, "val": va, "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}})
            stale = 0
        else:
            stale += 1
            if stale >= cfg.patience:
                break

    if best["state"] is not None:
        model.load_state_dict(best["state"])

    model.eval()
    with torch.no_grad():
        tr_pred = model(x_train)
        va_pred = model(x_val)
        te_pred = model(x_test)

    return {
        "train_mse": mse(tr_pred, y_train),
        "val_mse": mse(va_pred, y_val),
        "test_mse": mse(te_pred, y_test),
        "test_rel_err": relative_error(te_pred, y_test),
        "best_epoch": best["epoch"],
        "train_time_sec": time.time() - start,
        "curve": logs,
    }


def dump_curve(curve: dict, out_file: Path):
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(curve, f, indent=2)
