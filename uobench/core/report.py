"""Report generation for suites."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

from .diagnostics import compute
from .witness import verify
from ..io import load_instance


def summarize_instances(paths: Iterable[Path]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for inst_dir in paths:
        meta, arrays = load_instance(inst_dir)
        diag = compute(meta["id"], arrays)
        ok = verify(meta["id"], meta, arrays)
        row = {
            "id": meta["id"],
            "path": str(inst_dir),
            "feasible": "yes" if ok else "no",
        }
        for key, val in diag.items():
            row[key] = f"{val:.3e}"
        rows.append(row)
    return rows


def textual_overview(rows: List[Dict[str, str]]) -> str:
    if not rows:
        return "No instances available."
    by_id = defaultdict(list)
    for row in rows:
        by_id[row["id"]].append(row)
    lines = []
    for pid, items in by_id.items():
        lines.append(f"### {pid}")
        feas = sum(1 for r in items if r["feasible"] == "yes")
        lines.append(f"Feasibility: {feas}/{len(items)}")
        for key in items[0].keys():
            if key in {"id", "path", "feasible"}:
                continue
            vals = [float(r[key]) for r in items]
            lines.append(f"- {key}: min={min(vals):.2e}, median={sorted(vals)[len(vals)//2]:.2e}, max={max(vals):.2e}")
    return "\n".join(lines)


def write_markdown(md_path: Path, rows: List[Dict[str, str]]) -> None:
    md_path.parent.mkdir(parents=True, exist_ok=True)
    with md_path.open("w", encoding="utf-8") as fh:
        fh.write("# uobench Feasibility Summary\n\n")
        fh.write(textual_overview(rows))
        fh.write("\n")
        fh.write("\n| id | feasible | path |\n")
        fh.write("|---|---|---|\n")
        for row in rows:
            fh.write(f"| {row['id']} | {row['feasible']} | {row['path']} |\n")


def write_csv(csv_path: Path, rows: List[Dict[str, str]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row.keys()})
    with csv_path.open("w", encoding="utf-8") as fh:
        fh.write(",".join(keys) + "\n")
        for row in rows:
            fh.write(",".join(row.get(k, "" ) for k in keys) + "\n")

