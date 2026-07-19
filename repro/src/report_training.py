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
