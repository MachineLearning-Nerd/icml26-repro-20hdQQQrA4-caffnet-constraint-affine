#!/usr/bin/env python3
"""Independent checks for the exact-task named HardNet-Aff control audit."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
PAPER_ROOT = HERE.parents[1]
RESULTS = PAPER_ROOT / "outputs" / "hardnet_unicycle_results.json"
AUDIT_SOURCE = HERE / "audit_hardnet_unicycle_control.py"


def load_audit_module():
    spec = importlib.util.spec_from_file_location("hardnet_unicycle_audit", AUDIT_SOURCE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main():
    audit = load_audit_module()
    recomputed = audit.run_audit()
    committed = json.loads(RESULTS.read_text())
    exact_result_match = committed == recomputed

    # Re-evaluate the official HardNet-Aff one-sided equation in PyTorch rather
    # than trusting the NumPy implementation used by the primary audit.
    torch.set_default_dtype(torch.float64)
    rng = np.random.default_rng(20260722)
    maximum_torch_numpy_delta = 0.0
    for _ in range(256):
        a = rng.normal(size=(13, 2))
        b = rng.normal(size=13)
        raw = rng.normal(size=2)
        numpy_value = audit.hardnet_aff(raw, a, b)
        ta, tb, tf = map(torch.as_tensor, (a, b, raw))
        violation = torch.relu(ta @ tf - tb)
        torch_value = tf + torch.linalg.lstsq(ta, -violation).solution
        maximum_torch_numpy_delta = max(
            maximum_torch_numpy_delta,
            float(np.max(np.abs(numpy_value - torch_value.numpy()))),
        )

    main_run = recomputed["attempt_1_paper_x0"]
    checks = {
        "committed_result_exactly_recomputed": exact_result_match,
        "primary_audit_all_checks_passed": recomputed["all_checks_passed"],
        "official_torch_formula_matches_numpy": maximum_torch_numpy_delta < 2e-12,
        "exact_constraint_count": recomputed["paper_parameters"]["constraint_rows"] == 13,
        "exact_horizon": (
            recomputed["paper_parameters"]["dt"] == 0.1
            and recomputed["paper_parameters"]["steps"] == 150
        ),
        "hardnet_collision_pattern_exact": main_run["hardnet"]["collisions"] == [True, False, True],
        "caffnet_collision_pattern_exact": main_run["caffnet"]["collisions"] == [False, False, False],
        "scope_does_not_claim_goal_arrival": main_run["caffnet"]["final_goal_distance"] > 0.1,
        "hardnet_source_is_revision_pinned": len(recomputed["sources"]["hardnet_commit"]) == 40,
    }
    report = {
        "audit": "independent HardNet-Aff exact-task verification",
        "maximum_torch_numpy_delta": maximum_torch_numpy_delta,
        "checks": checks,
        "audit_passed": all(checks.values()),
        "primary_results_sha256": hashlib.sha256(RESULTS.read_bytes()).hexdigest(),
    }
    output = PAPER_ROOT / "outputs" / "hardnet_unicycle_audit.json"
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    output.write_text(serialized)
    print(serialized, end="")
    return 0 if report["audit_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
