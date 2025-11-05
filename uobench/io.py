"""Persistence utilities for benchmark instances and reports."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np


@dataclass
class InstancePaths:
    meta: Path
    data: Path
    readme: Path


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_instance(root: Path, suite: str, problem_id: str, scale: str, seed_tag: str,
                  meta: Dict, arrays: Dict[str, np.ndarray], readme: str) -> InstancePaths:
    inst_dir = root / f"{suite}_{scale}" / problem_id / seed_tag
    ensure_dir(inst_dir)
    meta_path = inst_dir / "meta.json"
    data_path = inst_dir / "data.npz"
    readme_path = inst_dir / "README.md"
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    np.savez_compressed(data_path, **arrays)
    with readme_path.open("w", encoding="utf-8") as fh:
        fh.write(readme)
    return InstancePaths(meta=meta_path, data=data_path, readme=readme_path)


def load_instance(path: str | Path) -> Tuple[Dict, Dict[str, np.ndarray]]:
    path = Path(path)
    with (path / "meta.json").open("r", encoding="utf-8") as fh:
        meta = json.load(fh)
    arrays = np.load(path / "data.npz")
    return meta, {k: arrays[k] for k in arrays.files}


def save_report(md_path: Path, csv_path: Path, rows: Iterable[Dict[str, str]]) -> None:
    ensure_dir(md_path.parent)
    ensure_dir(csv_path.parent)
    rows = list(rows)
    headers = rows[0].keys() if rows else []
    with csv_path.open("w", encoding="utf-8") as fh:
        if headers:
            fh.write(",".join(headers) + "\n")
        for row in rows:
            fh.write(",".join(str(row[h]) for h in headers) + "\n")
    with md_path.open("w", encoding="utf-8") as fh:
        fh.write("# uobench Feasibility Report\n\n")
        fh.write("| " + " | ".join(headers) + " |\n")
        fh.write("|" + "---|" * len(headers) + "\n")
        for row in rows:
            fh.write("| " + " | ".join(str(row[h]) for h in headers) + " |\n")


def load_suite_index(root: Path) -> Dict[str, Dict]:
    index: Dict[str, Dict] = {}
    for meta_path in root.rglob("meta.json"):
        with meta_path.open("r", encoding="utf-8") as fh:
            meta = json.load(fh)
        index[str(meta_path.parent)] = meta
    return index
