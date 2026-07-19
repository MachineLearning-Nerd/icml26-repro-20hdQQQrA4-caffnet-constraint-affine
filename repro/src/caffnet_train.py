#!/usr/bin/env python3
"""Trained-network experiments for CAffNet (arXiv 2605.24437, OpenReview 20hdQQQrA4).

The prior verdict scored C1 toy and C2/C3 inconclusive because no actual
neural network was trained. This module runs the paper's experiments with real
networks on CPU:

SCENARIO A - Section 4.1 / Appendix D.1 (exact spec): learn the piecewise
  target f: [-2,2] -> R under four input-dependent piecewise constraints
  (two upper, two lower; m=4 > n_out=1). 50 uniform training samples, 400
  linearly spaced test samples, MSE loss, Adam lr 1e-4, MLP 3x200 ReLU,
  50,000 epochs, batch 500 (full batch: only 50 samples), 5 seeds.

SCENARIO B - Section 4.2-style solver learning: minimize
  0.5 y'Qy + p'sin(y) s.t. Gy <= h, Cy = x with n_out=5, n_ineq=5, n_eq=3,
  h_i = sum_j |(G C^+)_ij| (Donti et al. feasibility), x ~ U[-1,1]^3,
  1000 train / 1000 test samples, unsupervised loss = objective value,
  10,000 epochs, batch 1000, 5 seeds. Benchmark optimizer: SciPy SLSQP
  multi-start (disclosed substitution for IPOPT).

METHODS (C2's exact wording - "fixed orthogonal or parallel projections"):
  - NN         unconstrained MLP + soft penalty 100*ReLU(Ay-b) (paper's setup)
  - OrthProj   FIXED orthogonal projection of the network output onto the
               violated constraints (least-squares correction, no learned
               null-space component), + soft penalty (m > n_out, so hard
               satisfaction is not guaranteed) - HardNet-style
  - ParProj    FIXED parallel projection: correct along a fixed direction
               (all-ones) instead of the orthogonal one, + soft penalty
  - CAffNet    the paper's full architecture: all sub-constraint combination
               projections P_gamma = f - A_g^+(A_g f - b_g) + (I-A_g^+A_g)w
               with a LEARNED null-space head w_phi (the joint-optimization
               ingredient), feasibility filtering, nearest-feasible selection
               (2-norm), returning f itself when already feasible.

CLAIM MAPPING:
  C1  trained CAffNet has exactly zero constraint violations at every train
      and test input in both scenarios (m > n_out, input-dependent b).
  C2  CAffNet (joint optimization incl. learned null space) vs the FIXED
      orthogonal/parallel projections: compare task loss and violations.
  C3  universal approximation preserved: width sweep {25,50,100,200,400} on
      Scenario A - test MSE decreases with capacity while violations remain
      exactly zero (plus the repo's existing exact adherence audit).

All runs are seeded and emit machine-readable JSON to outputs/.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs"
torch.set_num_threads(4)


# --------------------------------------------------------------- Scenario A
def target_f(x):
    return np.piecewise(
        x,
        [x <= -1, (x > -1) & (x <= 0), (x > 0) & (x <= 1), x > 1],
        [lambda t: -5 * np.sin(np.pi / 2 * (t + 1)) - 2,
         lambda t: -2.0 + 0 * t,
         lambda t: 2 - 9 * (t - 2 / 3) ** 2,
         lambda t: 3 / t ** 2 - 2])


def upper1(x):
    return np.piecewise(
        x,
        [x <= -1, (x > -1) & (x <= 0), (x > 0) & (x <= 1), x > 1],
        [lambda t: -3 * np.sin(np.pi / 2 * (t + 1)) + 1 / 5,
         lambda t: -2.0 + 0 * t,
         lambda t: 3 - 4 * (t - 1 / 2) ** 2,
         lambda t: 2.0 + 0 * t])


def upper2(x):
    return np.piecewise(
        x,
        [x <= -1, (x > -1) & (x <= 0), (x > 0) & (x <= 1), x > 1],
        [lambda t: -3 * np.sin(np.pi / 2 * (t + 1)) ** 3 + 1,
         lambda t: 2.0 + 0 * t,
         lambda t: 3 - 4 * (t - 4 / 5) ** 2,
         lambda t: 2.5 + 0 * t])


def lower1(x):
    return np.piecewise(
        x,
        [x <= -1, (x > -1) & (x <= 0), (x > 0) & (x <= 1), x > 1],
        [lambda t: 5 * np.sin(np.pi / 2 * (t + 1)) ** 2 - 3,
         lambda t: -2.0 + 0 * t,
         lambda t: (4 - 9 * (t - 2 / 3) ** 2) * t - 5 / 2,
         lambda t: 3 / t ** 3 - 5 / 2])


def lower2(x):
    return np.piecewise(
        x,
        [x <= -1, (x > -1) & (x <= 0), (x > 0) & (x <= 1), x > 1],
        [lambda t: 5 * np.sin(np.pi / 2 * (t + 1)) ** 8 - 2,
         lambda t: -3.0 + 0 * t,
         lambda t: (5 - 4 * (t - 1 / 6) ** 2) * t - 5 / 2,
         lambda t: 3 / (2 * t ** 3) - 16 / 9])


def scenario_a_bounds(x):
    """Return (lower, upper) as tight elementwise piecewise bounds."""
    lo = np.maximum(lower1(x), lower2(x))
    hi = np.minimum(upper1(x), upper2(x))
    return lo, hi


def check_scenario_a_feasible():
    x = np.linspace(-2, 2, 4001)
    lo, hi = scenario_a_bounds(x)
    f = target_f(x)
    return {
        "min_gap_upper_minus_lower": float((hi - lo).min()),
        "target_below_upper": bool((f <= hi + 1e-9).all()),
        "target_above_lower": bool((f >= lo - 1e-9).all()),
    }


class MLP(torch.nn.Module):
    def __init__(self, n_in, n_out, width=200, depth=3, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        layers, d = [], n_in
        for _ in range(depth):
            layers += [torch.nn.Linear(d, width), torch.nn.ReLU()]
            d = width
        layers.append(torch.nn.Linear(d, n_out))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def a_violation(y, lo, hi):
    """max over the four one-sided constraints of ReLU violation."""
    v = torch.cat([torch.relu(y - hi), torch.relu(lo - y)], dim=1)
    return v


def train_scenario_a(method, seed, width=200, epochs=50_000, log=print):
    rng = np.random.default_rng(seed)
    x_tr = rng.uniform(-2, 2, size=50)
    x_te = np.linspace(-2, 2, 400)
    t0 = time.time()

    def tensors(x):
        lo, hi = scenario_a_bounds(x)
        return (torch.tensor(x[:, None]), torch.tensor(target_f(x)[:, None]),
                torch.tensor(lo[:, None]), torch.tensor(hi[:, None]))

    X, Y, LO, HI = tensors(x_tr)
    Xt, Yt, LOt, HIt = tensors(x_te)
    f_net = MLP(1, 1, width=width, seed=seed).double()
    params = list(f_net.parameters())
    opt = torch.optim.Adam(params, lr=1e-4)

    def forward(x, lo, hi):
        f = f_net(x)
        if method == "NN":
            return f
        if method == "CAffNet":
            # candidates: f itself, plus the k=1 sub-constraint projections
            # (in 1-D these are exactly the four boundary values); nearest
            # feasible candidate wins when f is infeasible = clamp to [lo,hi]
            return torch.clamp(f, lo, hi)
        raise ValueError(method)

    for _ in range(epochs):
        opt.zero_grad()
        y = forward(X, LO, HI)
        loss = ((y - Y) ** 2).mean()
        if method == "NN":
            loss = loss + 100.0 * a_violation(y, LO, HI).mean()
        loss.backward()
        opt.step()
    train_s = time.time() - t0

    with torch.no_grad():
        def evaluate(x, y_true, lo, hi):
            f = f_net(x)
            if method == "CAffNet":
                y = torch.clamp(f, lo, hi)
            else:
                y = f
            v = a_violation(y, lo, hi)
            return (float(((y - y_true) ** 2).mean()),
                    float(v.max()), float(v.mean()))

        mse_te, vmax_te, vmean_te = evaluate(Xt, Yt, LOt, HIt)
        mse_tr, vmax_tr, vmean_tr = evaluate(X, Y, LO, HI)
    return {
        "method": method, "seed": seed, "width": width,
        "train_seconds": round(train_s, 1),
        "test_mse": mse_te, "train_mse": mse_tr,
        "test_violation_max": vmax_te, "test_violation_mean": vmean_te,
        "train_violation_max": vmax_tr,
    }


# --------------------------------------------------------------- Scenario B
class ScenarioB:
    """Random solver-learning instance, Donti-style guaranteed-feasible h.

    With the paper's h = sum_j |(G C^+)_ij| the inequalities never bind for
    this instance family (verified in outputs), so `h_scale < 1` provides a
    TIGHTENED variant: h' = h_scale * h + guard, with the guard chosen by an
    exact LP feasibility check so that {y: Gy <= h', Cy = x} remains nonempty
    for every x in [-1,1]^n_eq (checked at all corners; the feasibility
    function is convex in x, so corners suffice). The tightened variant makes
    inequality combinations genuinely active, which is what distinguishes the
    learned null-space (joint optimization) from fixed projections."""

    def __init__(self, seed, n_out=5, n_ineq=5, n_eq=3, h_scale=1.0):
        rng = np.random.default_rng(1000 + seed)
        self.Q = np.diag(rng.uniform(0.5, 2.0, n_out))
        self.p = rng.uniform(-1.0, 1.0, n_out)
        self.G = rng.normal(size=(n_ineq, n_out))
        self.C = rng.normal(size=(n_eq, n_out))
        GCp = self.G @ np.linalg.pinv(self.C)
        self.h = np.abs(GCp).sum(axis=1)
        self.n_out, self.n_ineq, self.n_eq = n_out, n_ineq, n_eq
        if h_scale != 1.0:
            self.h = self._tighten(h_scale)
        self.h_scale = h_scale

    def _min_feasible_slack(self, h):
        """max over corners x of min over y of max_i (Gy - h)_i s.t. Cy=x,
        computed exactly with scipy linprog (epigraph form)."""
        from itertools import product as iproduct
        from scipy.optimize import linprog
        worst = -np.inf
        for corner in iproduct((-1.0, 1.0), repeat=self.n_eq):
            x = np.array(corner)
            # variables: y (n_out), s (1); minimize s
            c = np.zeros(self.n_out + 1); c[-1] = 1.0
            A_ub = np.hstack([self.G, -np.ones((self.n_ineq, 1))])
            b_ub = h.copy()
            A_eq = np.hstack([self.C, np.zeros((self.n_eq, 1))])
            r = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=x,
                        bounds=[(None, None)] * (self.n_out + 1),
                        method="highs")
            if r.status == 3:      # unbounded below => strictly feasible
                continue
            assert r.status == 0, "feasibility LP failed"
            worst = max(worst, r.fun)
        return worst   # <= 0 means feasible for all corners

    def _tighten(self, scale):
        h = scale * self.h
        slack = self._min_feasible_slack(h)
        if slack > 0:            # push h out just enough plus 5% margin
            h = h + (slack * 1.05)
        assert self._min_feasible_slack(h) <= 1e-9
        return h

    def objective_t(self, y):
        Q = torch.tensor(self.Q)
        p = torch.tensor(self.p)
        return 0.5 * (y @ Q * y).sum(dim=1) + (torch.sin(y) * p).sum(dim=1)

    def violations_t(self, y, x):
        G = torch.tensor(self.G)
        C = torch.tensor(self.C)
        h = torch.tensor(self.h)
        ineq = torch.relu(y @ G.T - h)
        eq = (y @ C.T - x).abs()
        return ineq, eq

    def combos(self):
        """Sub-constraint combinations: all 3 equalities always active,
        plus 0..2 inequality rows (k = n_eq + i <= n_out)."""
        from itertools import combinations
        out = []
        for i in range(0, self.n_out - self.n_eq + 1):
            for c in combinations(range(self.n_ineq), i):
                out.append(tuple(c))
        return out


class CaffnetB(torch.nn.Module):
    """CAffNet for Scenario B: f-head + learned null-space head w_phi."""

    def __init__(self, inst: ScenarioB, seed, width=200):
        super().__init__()
        self.inst = inst
        self.f_net = MLP(inst.n_eq, inst.n_out, width=width, seed=seed)
        self.w_net = MLP(inst.n_eq, inst.n_out, width=width, seed=seed + 500)
        combos = inst.combos()
        pinvs, projs, sel = [], [], []
        for c in combos:
            A = (np.vstack([inst.C, inst.G[list(c)]]) if c
                 else inst.C.copy())
            Ap = np.linalg.pinv(A)
            pinvs.append(Ap)
            projs.append(np.eye(inst.n_out) - Ap @ A)
            sel.append(c)
        self.sel = sel
        self.Ap_all = [torch.tensor(a) for a in pinvs]
        self.Pn_all = [torch.tensor(a) for a in projs]

    def forward(self, x):
        inst = self.inst
        f = self.f_net(x)
        w = self.w_net(x)
        G = torch.tensor(inst.G)
        C = torch.tensor(inst.C)
        h = torch.tensor(inst.h)
        cands = []
        for c, Ap, Pn in zip(self.sel, self.Ap_all, self.Pn_all):
            b = torch.cat(
                [x] + ([h[list(c)].expand(x.shape[0], len(c))] if c else []),
                dim=1)
            A = torch.cat([C] + ([G[list(c)]] if c else []), dim=0)
            P = f - (f @ A.T - b) @ Ap.T + w @ Pn.T
            cands.append(P)
        cands = torch.stack(cands, dim=1)              # (B, n_combo, n_out)
        ineq = torch.relu(cands @ G.T - h)             # (B, n_combo, n_ineq)
        eq = (cands @ C.T - x.unsqueeze(1)).abs()
        feas = (ineq.max(dim=2).values <= 1e-9) & (eq.max(dim=2).values <= 1e-7)
        dist = ((cands - f.unsqueeze(1)) ** 2).sum(dim=2)
        dist = torch.where(feas, dist, torch.full_like(dist, torch.inf))
        idx = dist.argmin(dim=1)
        y = cands[torch.arange(x.shape[0]), idx]
        # if f itself satisfies everything, keep f (paper eq. (6))
        f_ineq = torch.relu(f @ G.T - h).max(dim=1).values
        f_eq = (f @ C.T - x).abs().max(dim=1).values
        f_ok = (f_ineq <= 1e-9) & (f_eq <= 1e-7)
        return torch.where(f_ok.unsqueeze(1), f, y), feas.any(dim=1)


def fixed_projection_b(inst, f, x, kind):
    """Fixed (non-learned) projections for the C2 baselines."""
    C = torch.tensor(inst.C)
    Cp = torch.tensor(np.linalg.pinv(inst.C))
    if kind == "orth":
        # orthogonal least-squares restoration of the equality constraints
        return f - (f @ C.T - x) @ Cp.T
    if kind == "par":
        # parallel projection along the fixed all-ones direction
        d = torch.ones(inst.n_out, dtype=torch.float64)
        Cd = C @ d                                     # (n_eq,)
        M = torch.tensor(np.linalg.pinv((inst.C @ d.numpy())[:, None]))
        return f - torch.outer(((f @ C.T - x) @ M.T).squeeze(1), d)
    raise ValueError(kind)


def slsqp_benchmark(inst, xs, starts=3, seed=0):
    from scipy.optimize import minimize
    rng = np.random.default_rng(seed)
    vals = []
    for x in xs:
        best = np.inf
        for s in range(starts):
            y0 = np.linalg.pinv(inst.C) @ x + \
                (rng.normal(scale=0.3, size=inst.n_out) if s else 0)
            r = minimize(
                lambda y: 0.5 * y @ inst.Q @ y + inst.p @ np.sin(y),
                y0, method="SLSQP",
                constraints=[
                    {"type": "eq", "fun": lambda y: inst.C @ y - x},
                    {"type": "ineq", "fun": lambda y: inst.h - inst.G @ y}],
                options={"maxiter": 200, "ftol": 1e-10})
            if r.success and r.fun < best:
                best = r.fun
        vals.append(best)
    return np.array(vals)


def train_scenario_b(method, seed, epochs=10_000, log=print, h_scale=1.0):
    inst = ScenarioB(0, h_scale=h_scale)   # shared instance across methods
    rng = np.random.default_rng(seed)
    x_tr = rng.uniform(-1, 1, size=(1000, inst.n_eq))
    x_te = np.random.default_rng(4242).uniform(-1, 1, size=(1000, inst.n_eq))
    X, Xt = torch.tensor(x_tr), torch.tensor(x_te)
    t0 = time.time()

    if method == "CAffNet":
        model = CaffnetB(inst, seed).double()
        params = list(model.parameters())
    else:
        f_net = MLP(inst.n_eq, inst.n_out, seed=seed).double()
        params = list(f_net.parameters())
    opt = torch.optim.Adam(params, lr=1e-4)

    def outputs(x):
        if method == "CAffNet":
            y, _ = model(x)
            return y
        f = f_net(x)
        if method == "NN":
            return f
        if method == "OrthProj":
            return fixed_projection_b(inst, f, x, "orth")
        if method == "ParProj":
            return fixed_projection_b(inst, f, x, "par")
        raise ValueError(method)

    for _ in range(epochs):
        opt.zero_grad()
        y = outputs(X)
        loss = inst.objective_t(y).mean()
        if method != "CAffNet":
            ineq, eq = inst.violations_t(y, X)
            loss = loss + 100.0 * (ineq.mean() + eq.mean())
        loss.backward()
        opt.step()
    train_s = time.time() - t0

    with torch.no_grad():
        y = outputs(Xt)
        obj = float(inst.objective_t(y).mean())
        ineq, eq = inst.violations_t(y, Xt)
        res = {
            "method": method, "seed": seed,
            "train_seconds": round(train_s, 1),
            "test_objective": obj,
            "ineq_violation_max": float(ineq.max()),
            "ineq_violation_mean": float(ineq.mean()),
            "eq_violation_max": float(eq.max()),
            "eq_violation_mean": float(eq.mean()),
        }
    return res


# ------------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["a", "b", "btight", "widths", "bench", "all"],
                    default="all")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs-a", type=int, default=50_000)
    ap.add_argument("--epochs-b", type=int, default=10_000)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    report = {}
    feas = check_scenario_a_feasible()
    print("scenario A constraint audit:", json.dumps(feas), flush=True)
    assert feas["target_below_upper"] and feas["target_above_lower"], \
        "Appendix D.1 reconstruction must bracket the target"
    report["scenario_a_spec_audit"] = feas

    if args.part in ("a", "all"):
        rows = []
        # In 1-D every fixed projection reduces to clamping, i.e. coincides
        # with CAffNet's candidate selection, so the projection baselines are
        # only meaningful in Scenario B (n_out=5). Here we compare the
        # paper's NN-with-soft-penalty against the full CAffNet.
        for method in ("NN", "CAffNet"):
            for seed in range(args.seeds):
                r = train_scenario_a(method, seed, epochs=args.epochs_a)
                rows.append(r)
                print("[A]", json.dumps(r), flush=True)
        report["scenario_a"] = rows
    if args.part in ("widths", "all"):
        rows = []
        for width in (25, 50, 100, 200, 400):
            r = train_scenario_a("CAffNet", 0, width=width,
                                 epochs=args.epochs_a)
            rows.append(r)
            print("[W]", json.dumps(r), flush=True)
        report["width_sweep"] = rows
    if args.part in ("b", "all"):
        rows = []
        for method in ("NN", "OrthProj", "ParProj", "CAffNet"):
            for seed in range(args.seeds):
                r = train_scenario_b(method, seed, epochs=args.epochs_b)
                rows.append(r)
                print("[B]", json.dumps(r), flush=True)
        report["scenario_b"] = rows
    if args.part in ("btight", "all"):
        rows = []
        for method in ("NN", "OrthProj", "ParProj", "CAffNet"):
            for seed in range(args.seeds):
                r = train_scenario_b(method, seed, epochs=args.epochs_b,
                                     h_scale=0.25)
                r["h_scale"] = 0.25
                rows.append(r)
                print("[Btight]", json.dumps(r), flush=True)
        report["scenario_b_tight"] = rows
    if args.part in ("bench", "all"):
        inst = ScenarioB(0)
        xs = np.random.default_rng(4242).uniform(-1, 1, size=(200, inst.n_eq))
        vals = slsqp_benchmark(inst, xs)
        report["scenario_b_benchmark"] = {
            "optimizer": "SciPy SLSQP multi-start (disclosed IPOPT substitute)",
            "n_points": len(xs),
            "mean_objective": float(np.mean(vals)),
        }
        print("[bench]", json.dumps(report["scenario_b_benchmark"]), flush=True)

    stamp = args.part
    path = OUT / f"caffnet_train_{stamp}.json"
    existing = {}
    full = OUT / "caffnet_train_report.json"
    if full.exists():
        existing = json.loads(full.read_text())
    existing.update(report)
    full.write_text(json.dumps(existing, indent=2) + "\n")
    path.write_text(json.dumps(report, indent=2) + "\n")
    print("wrote", full.name)


if __name__ == "__main__":
    main()
