"""Tests for the end-to-end verification routines."""

from __future__ import annotations
import json
from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

pytest.importorskip("numpy")

from uobgen.io import save_instance
from uobgen.problems.a_smooth import generate_a1_qp
from uobgen.problems.b_nonsmooth import generate_b1_lasso, generate_b3_svm
from uobgen.problems.c_semismooth import generate_c1_vi, generate_c2_lcp
from uobgen.problems.d_conic import generate_d1_socp
from uobgen.verification import verify_instance


def _write_instance(tmp_path: Path, instance: dict, name: str) -> Path:
    inst_dir = save_instance(tmp_path, instance, name=name)
    meta_path = inst_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta.setdefault("meta", {})["scale"] = "S"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return inst_dir


def test_qp_solver_residual(tmp_path: Path) -> None:
    inst = generate_a1_qp(seed=7, n=15, kappa=1e2)
    path = _write_instance(tmp_path, inst, "A1")
    report = verify_instance(path)
    assert report.properties.ok
    assert report.solver.attempted and report.solver.ok
    assert report.solver.residual is not None and report.solver.residual < 1e-6


def test_lasso_solver_converges(tmp_path: Path) -> None:
    inst = generate_b1_lasso(seed=11, m=120, n=80, sparsity=0.1, rho=0.2)
    path = _write_instance(tmp_path, inst, "B1")
    report = verify_instance(path)
    assert report.properties.ok
    assert report.solver.attempted and report.solver.ok
    assert report.solver.residual is not None and report.solver.residual < 1e-3


def test_svm_solver_converges(tmp_path: Path) -> None:
    inst = generate_b3_svm(seed=5, m=200, n=40, mu_norm=1.5)
    path = _write_instance(tmp_path, inst, "B3")
    report = verify_instance(path)
    assert report.solver.attempted and report.solver.ok


def test_vi_and_lcp_solvers(tmp_path: Path) -> None:
    vi_inst = generate_c1_vi(seed=3, n=20, mu=0.1, cond=10.0)
    vi_path = _write_instance(tmp_path, vi_inst, "C1")
    vi_report = verify_instance(vi_path)
    assert vi_report.properties.ok
    assert vi_report.solver.attempted and vi_report.solver.ok

    lcp_inst = generate_c2_lcp(seed=9, n=12, delta=1e-2)
    lcp_path = _write_instance(tmp_path, lcp_inst, "C2")
    lcp_report = verify_instance(lcp_path)
    assert lcp_report.properties.ok
    assert lcp_report.solver.attempted and lcp_report.solver.ok


def test_socp_solver(tmp_path: Path) -> None:
    inst = generate_d1_socp(seed=4, m=150, n=60, R=5.0)
    path = _write_instance(tmp_path, inst, "D1")
    report = verify_instance(path)
    assert report.properties.ok
    assert report.solver.attempted and report.solver.ok
