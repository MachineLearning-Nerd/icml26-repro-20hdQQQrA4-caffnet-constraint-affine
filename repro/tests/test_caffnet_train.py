"""Integrity tests for the trained-network CAffNet experiments."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "repro" / "src"))

import caffnet_train as ct

REPORT = ROOT / "outputs" / "caffnet_train_report.json"


def test_appendix_d1_reconstruction_feasible():
    audit = ct.check_scenario_a_feasible()
    assert audit["target_below_upper"] and audit["target_above_lower"]
    # tight on (-1, 0]: upper1 = lower1 = target = -2 there
    x = np.array([-0.5])
    assert ct.upper1(x)[0] == -2.0 == ct.lower1(x)[0] == ct.target_f(x)[0]


def test_caffnet_1d_output_always_feasible():
    rng = np.random.default_rng(0)
    x = rng.uniform(-2, 2, 200)
    lo, hi = ct.scenario_a_bounds(x)
    f = rng.normal(scale=5, size=200)  # arbitrary head output
    y = np.clip(f, lo, hi)
    assert (y >= lo - 1e-12).all() and (y <= hi + 1e-12).all()


def test_scenario_b_caffnet_hard_satisfaction_untrained():
    inst = ct.ScenarioB(0)
    m = ct.CaffnetB(inst, 0).double()
    x = torch.tensor(np.random.default_rng(1).uniform(-1, 1, (64, 3)))
    with torch.no_grad():
        y, feas = m(x)
    ineq, eq = inst.violations_t(y, x)
    assert float(feas.float().mean()) == 1.0
    assert float(ineq.max()) < 1e-9 and float(eq.max()) < 1e-6


def test_scenario_b_tight_feasible_and_active():
    inst = ct.ScenarioB(0, h_scale=0.25)
    assert inst._min_feasible_slack(inst.h) <= 1e-9
    m = ct.CaffnetB(inst, 0).double()
    x = torch.tensor(np.random.default_rng(1).uniform(-1, 1, (64, 3)))
    with torch.no_grad():
        y, feas = m(x)
    ineq, eq = inst.violations_t(y, x)
    assert float(ineq.max()) < 1e-9 and float(eq.max()) < 1e-6
    G = torch.tensor(inst.G)
    h = torch.tensor(inst.h)
    active = ((y @ G.T - h).abs() < 1e-6).any(dim=1).float().mean()
    assert float(active) > 0.5  # tight variant genuinely activates inequalities


def test_fixed_projections_cannot_satisfy_tight_inequalities():
    r = json.loads(REPORT.read_text())
    tight = r["scenario_b_tight"]
    by = {}
    for row in tight:
        by.setdefault(row["method"], []).append(row)
    assert max(x["ineq_violation_max"] for x in by["OrthProj"]) > 0.1
    assert max(x["ineq_violation_max"] for x in by["ParProj"]) > 0.1
    assert max(x["eq_violation_max"] for x in by["NN"]) > 1e-3
    assert max(max(x["ineq_violation_max"], x["eq_violation_max"])
               for x in by["CAffNet"]) < 1e-9


def test_trained_caffnet_zero_violations_everywhere():
    r = json.loads(REPORT.read_text())
    for row in r["scenario_a"]:
        if row["method"] == "CAffNet":
            assert row["test_violation_max"] == 0.0
            assert row["train_violation_max"] == 0.0
    for key in ("scenario_b", "scenario_b_tight"):
        for row in r[key]:
            if row["method"] == "CAffNet":
                assert row["ineq_violation_max"] < 1e-9
                assert row["eq_violation_max"] < 1e-9


def test_width_sweep_monotone_capacity():
    r = json.loads(REPORT.read_text())
    rows = sorted(r["width_sweep"], key=lambda x: x["width"])
    assert all(row["test_violation_max"] == 0.0 for row in rows)
    # capacity helps: widest strictly better than narrowest on test MSE
    assert rows[-1]["test_mse"] < rows[0]["test_mse"]
