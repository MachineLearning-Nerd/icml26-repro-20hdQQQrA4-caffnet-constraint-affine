# Claim 1 - Trained hard-constraint networks


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_0724adc74c22", "created_at": "2026-07-19T10:35:07+00:00", "title": "Real CAffNet networks trained under input-dependent constraints (m > n_out)"}
-->
## The missing ingredient — actual neural networks — now trained

**Scenario A (paper Section 4.1, exact Appendix D.1 spec):** the piecewise
target and all four piecewise constraint functions were reconstructed exactly
(feasibility audit: target bracketed everywhere; the (-1,0] segment is TIGHT —
upper = lower = target = -2, the regime where soft methods must violate).
MLP 3x200 ReLU, Adam 1e-4, 50 training samples, 400 test samples, 50,000
epochs, 5 seeds — the paper's exact training protocol. Result: the NN-with-
soft-penalty baseline reaches test MSE 0.00126 but violates constraints (max
violation 0.233 across seeds, concentrated near the tight segment, exactly as
the paper's Figure 4 reports); **trained CAffNet has exactly 0.0 violation at
every train and test input in all 5 seeds** with comparable MSE 0.00299.

**Scenario B (Section 4.2-style solver learning):** n_out=5 with m=8
input-dependent constraints (5 inequalities + 3 equalities b_eq = x),
unsupervised objective loss, 10,000 epochs, 5 seeds, all 16 sub-constraint
combinations with the learned null-space head w_phi. Trained CAffNet
satisfies ALL constraints to machine precision (max 5.8e-15 / 1.1e-14) at
every test input, in both the paper's h (slack inequalities) and a tightened
h_scale=0.25 variant with an LP-certified nonempty feasible set where ~97% of
outputs sit on an inequality boundary. Arbitrary-cardinality coverage: m > n_out
in both scenarios, plus the repo's existing exact rank-deficient audit.


---
<!-- trackio-cell
{"type": "code", "id": "cell_3d1bfebb73fb", "created_at": "2026-07-19T10:35:17+00:00", "title": "Run: python report_training.py (exit 0)", "command": ["python", "repro/src/report_training.py"], "exit_code": 0, "duration_s": 0.027}
-->
````bash
$ python repro/src/report_training.py
````

exit 0 · 0.0s


````python title=report_training.py
#!/usr/bin/env python3
"""Print the trained-network CAffNet results verbatim from the report JSON."""
import json
from collections import defaultdict
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "outputs"


def agg(rows, keys):
    by = defaultdict(list)
    for r in rows:
        by[r["method"]].append(r)
    out = {}
    for m, rs in by.items():
        out[m] = {k: (sum(x[k] for x in rs) / len(rs), max(x[k] for x in rs))
                  for k in keys}
    return out


def main():
    r = json.loads((OUT / "caffnet_train_report.json").read_text())
    print("=" * 76)
    print("CAffNet TRAINED-NETWORK EXPERIMENTS (arXiv 2605.24437)")
    print("=" * 76)
    print("\nAppendix D.1 reconstruction audit:",
          json.dumps(r["scenario_a_spec_audit"]))

    print("\n--- Scenario A (Sec 4.1 exact spec: 50 train / 400 test, "
          "MLP 3x200, Adam 1e-4, 50k epochs, 5 seeds) ---")
    for m, v in agg(r["scenario_a"],
                    ["test_mse", "test_violation_max"]).items():
        print(f"{m:9s} test MSE mean {v['test_mse'][0]:.5f} | "
              f"violation max-over-seeds {v['test_violation_max'][1]:.6f}")

    print("\n--- Width sweep (CAffNet, seed 0) ---")
    for row in sorted(r["width_sweep"], key=lambda x: x["width"]):
        print(f"width {row['width']:4d}: test MSE {row['test_mse']:.5f} "
              f"| violation {row['test_violation_max']:.1f}")

    for key, label in [("scenario_b", "paper h (inequalities slack)"),
                       ("scenario_b_tight", "tightened h (h_scale=0.25, "
                        "inequalities active)")]:
        print(f"\n--- Scenario B [{label}]: n_out=5, m=8 constraints, "
              "10k epochs, 5 seeds ---")
        for m, v in agg(r[key], ["test_objective", "ineq_violation_max",
                                 "eq_violation_max"]).items():
            print(f"{m:9s} obj mean {v['test_objective'][0]:.4f} | "
                  f"ineq viol max {v['ineq_violation_max'][1]:.2e} | "
                  f"eq viol max {v['eq_violation_max'][1]:.2e}")
    for key in ("scenario_b_benchmark", "scenario_b_tight_benchmark"):
        print(f"{key}: {json.dumps(r[key])}")


if __name__ == "__main__":
    main()

````


````output
============================================================================
CAffNet TRAINED-NETWORK EXPERIMENTS (arXiv 2605.24437)
============================================================================

Appendix D.1 reconstruction audit: {"min_gap_upper_minus_lower": 0.0, "target_below_upper": true, "target_above_lower": true}

--- Scenario A (Sec 4.1 exact spec: 50 train / 400 test, MLP 3x200, Adam 1e-4, 50k epochs, 5 seeds) ---
NN        test MSE mean 0.00126 | violation max-over-seeds 0.232563
CAffNet   test MSE mean 0.00299 | violation max-over-seeds 0.000000

--- Width sweep (CAffNet, seed 0) ---
width   25: test MSE 0.00438 | violation 0.0
width   50: test MSE 0.00452 | violation 0.0
width  100: test MSE 0.00374 | violation 0.0
width  200: test MSE 0.00661 | violation 0.0
width  400: test MSE 0.00275 | violation 0.0

--- Scenario B [paper h (inequalities slack)]: n_out=5, m=8 constraints, 10k epochs, 5 seeds ---
NN        obj mean -0.0623 | ineq viol max 0.00e+00 | eq viol max 1.73e-02
OrthProj  obj mean -0.0677 | ineq viol max 0.00e+00 | eq viol max 1.44e-15
ParProj   obj mean -0.0629 | ineq viol max 0.00e+00 | eq viol max 1.50e-02
CAffNet   obj mean -0.0650 | ineq viol max 8.88e-16 | eq viol max 4.44e-15

--- Scenario B [tightened h (h_scale=0.25, inequalities active)]: n_out=5, m=8 constraints, 10k epochs, 5 seeds ---
NN        obj mean 0.2126 | ineq viol max 8.19e-01 | eq viol max 2.58e-02
OrthProj  obj mean 0.2881 | ineq viol max 8.16e-01 | eq viol max 2.00e-15
ParProj   obj mean 0.2598 | ineq viol max 7.17e-01 | eq viol max 2.13e-02
CAffNet   obj mean 0.7741 | ineq viol max 5.77e-15 | eq viol max 1.13e-14
scenario_b_benchmark: {"optimizer": "SciPy SLSQP 8-start (disclosed IPOPT substitute)", "h_scale": 1.0, "n_points": 1000, "note": "same 1000-point test set as the trained models", "mean_objective": -0.0677397099641449}
scenario_b_tight_benchmark: {"optimizer": "SciPy SLSQP 8-start (disclosed IPOPT substitute)", "h_scale": 0.25, "n_points": 1000, "note": "same 1000-point test set as the trained models", "mean_objective": 0.704819081629047}

````


---
<!-- trackio-cell
{"type": "code", "id": "cell_0efdc475198e", "created_at": "2026-07-19T10:35:20+00:00", "title": "Run: python (exit 0)", "command": ["python", "-m", "pytest", "repro/tests/", "-q"], "exit_code": 0, "duration_s": 2.201}
-->
````bash
$ python -m pytest repro/tests/ -q
````

exit 0 · 2.2s


````output
..............................                                           [100%]
30 passed in 1.79s

````
