"""Input/output helpers for UOBGen instances."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np


def save_instance(base_path: Path | str, instance: Dict[str, Any], name: str | None = None) -> Path:
    """Save an instance to ``base_path``.

    Parameters
    ----------
    base_path:
        Directory where the instance folder will be created.
    instance:
        Dictionary returned by problem generators. Must contain ``id``, ``meta`` and ``data`` keys.

    Returns
    -------
    Path
        Path to the created instance directory.
    """
    path = Path(base_path)
    path.mkdir(parents=True, exist_ok=True)
    dir_name = name or instance['id']
    instance_dir = path / dir_name
    instance_dir.mkdir(parents=True, exist_ok=True)

    reference = instance.get("reference", {}) or {}
    reference_payload = {}
    for key, value in reference.items():
        if value is None:
            continue
        reference_payload[key] = np.asarray(value)

    meta = {
        "id": instance["id"],
        "name": instance.get("name"),
        "seed": instance.get("seed"),
        "meta": instance.get("meta", {}),
        "reference": {
            "has_reference": bool(reference_payload),
            "fields": sorted(reference_payload.keys()),
        },
    }
    (instance_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    np.savez_compressed(instance_dir / "data.npz", **instance.get("data", {}))
    if reference_payload:
        np.savez_compressed(instance_dir / "reference.npz", **reference_payload)

    readme_lines = [
        f"# {instance.get('name', instance['id'])}",
        "",
        "## Parameters",
        json.dumps(instance.get("meta", {}).get("knobs", {}), indent=2, ensure_ascii=False),
        "",
        "## Diagnostics",
        json.dumps(instance.get("meta", {}).get("diagnostics", {}), indent=2, ensure_ascii=False),
    ]
    (instance_dir / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")
    return instance_dir


def load_instance(path: Path | str) -> Dict[str, Any]:
    """Load an instance saved by :func:`save_instance`."""
    instance_dir = Path(path)
    meta = json.loads((instance_dir / "meta.json").read_text(encoding="utf-8"))
    data = np.load(instance_dir / "data.npz")
    arrays = {k: data[k] for k in data.files}
    references = {}
    ref_path = instance_dir / "reference.npz"
    if ref_path.exists():
        ref = np.load(ref_path)
        references = {k: ref[k] for k in ref.files}
    readme = (instance_dir / "README.md").read_text(encoding="utf-8")
    return {"meta": meta, "data": arrays, "readme": readme, "reference": references}
