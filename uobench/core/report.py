"""Reporting utilities for suite-wide feasibility checks."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List

from .diagnostics import compute
from .witness import verify
from ..io import load_instance


def summarize_instances(paths: Iterable[Path]) -> List[Dict[str, str]]:
    """Load instances and build per-instance diagnostic rows."""

    rows: List[Dict[str, str]] = []
    for inst_dir in paths:
        meta, arrays = load_instance(inst_dir)
        diag = meta.get("diagnostics") or compute(meta["id"], arrays)
        ok = verify(meta["id"], meta, arrays)
        row: Dict[str, str] = {
            "id": meta["id"],
            "family": meta.get("family", "unknown"),
            "path": str(inst_dir),
            "feasible": "yes" if ok else "no",
            "seed": str(meta.get("seed", "")),
        }
        for dim_name, dim_val in (meta.get("dims") or {}).items():
            row[f"dim_{dim_name}"] = str(dim_val)
        for knob_name, knob_val in (meta.get("knobs") or {}).items():
            row[f"knob_{knob_name}"] = str(knob_val)
        witness = meta.get("witness") or {}
        if witness:
            row["witness"] = witness.get("cert_type", "provided")
        for key, val in diag.items():
            row[f"diag_{key}"] = f"{val:.5e}"
        rows.append(row)
    return rows


def write_markdown(md_path: Path, rows: List[Dict[str, str]]) -> None:
    """Create a Markdown report summarising feasibility and diagnostics."""

    md_path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted(
        {
            k
            for row in rows
            for k in row.keys()
            if k not in {"id", "family", "path", "feasible"}
        }
    )
    with md_path.open("w", encoding="utf-8") as fh:
        fh.write("# uobench Feasibility Report\n\n")
        by_problem: Dict[str, List[Dict[str, str]]] = defaultdict(list)
        for row in rows:
            by_problem[row["id"]].append(row)
        for pid, plist in sorted(by_problem.items()):
            fh.write(f"## {pid}\n")
            feas = sum(1 for r in plist if r["feasible"] == "yes")
            fh.write(f"Feasibility: {feas}/{len(plist)} instances verified.\n\n")
            for key in keys:
                numeric_vals = []
                for r in plist:
                    if key not in r:
                        continue
                    try:
                        numeric_vals.append(float(r[key]))
                    except (TypeError, ValueError):
                        continue
                if not numeric_vals:
                    continue
                fh.write(
                    f"- {key}: min={min(numeric_vals):.3e}, median={median(numeric_vals):.3e}, max={max(numeric_vals):.3e}\n"
                )
            fh.write("\n")
        fh.write("\n### Instance table\n")
        headers = ["id", "feasible", "path", *keys]
        fh.write("| " + " | ".join(headers) + " |\n")
        fh.write("|" + " --- |" * len(headers) + "\n")
        for row in rows:
            fh.write("| " + " | ".join(row.get(h, "") for h in headers) + " |\n")


def write_csv(csv_path: Path, rows: List[Dict[str, str]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row.keys()})
    with csv_path.open("w", encoding="utf-8") as fh:
        fh.write(",".join(keys) + "\n")
        for row in rows:
            fh.write(",".join(row.get(k, "") for k in keys) + "\n")


def write_json(json_path: Path, rows: List[Dict[str, str]]) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)
