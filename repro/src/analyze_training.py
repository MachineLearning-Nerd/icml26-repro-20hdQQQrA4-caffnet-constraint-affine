"""Validate and summarize the executed CAffNet neural experiments."""
from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs"


def _load(name: str) -> dict:
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def _sha256(name: str) -> str:
    return hashlib.sha256((OUT / name).read_bytes()).hexdigest()


def _method(rows: list[dict], name: str) -> list[dict]:
    selected = [row for row in rows if row["method"] == name]
    if {row["seed"] for row in selected} != set(range(5)):
        raise ValueError(f"{name} must contain exactly seeds 0..4")
    return selected


def _mean(rows: list[dict], key: str) -> float:
    return statistics.mean(float(row[key]) for row in rows)


def analyze() -> dict:
    a = _load("caffnet_train_a.json")
    b = _load("caffnet_train_b.json")
    b_tight = _load("caffnet_train_btight.json")
    widths = _load("caffnet_train_widths.json")
    joint = _load("joint_control/summary.json")

    a_nn = _method(a["scenario_a"], "NN")
    a_caff = _method(a["scenario_a"], "CAffNet")
    if {row["width"] for row in a["scenario_a"]} != {200}:
        raise ValueError("Scenario A must use the paper's width 200")
    b_methods = {name: _method(b["scenario_b"], name) for name in ("NN", "OrthProj", "ParProj", "CAffNet")}
    tight_methods = {
        name: _method(b_tight["scenario_b_tight"], name)
        for name in ("NN", "OrthProj", "ParProj", "CAffNet")
    }
    if {row["h_scale"] for row in b_tight["scenario_b_tight"]} != {0.25}:
        raise ValueError("tight stress control must use h_scale=0.25")
    sweep = widths["width_sweep"]
    if [row["width"] for row in sweep] != [25, 50, 100, 200, 400]:
        raise ValueError("unexpected width sweep")

    paper_ff_mse = 0.0020
    paper_ff_std = 0.0032
    caff_mse = _mean(a_caff, "test_mse")
    analysis = {
        "paper": "20hdQQQrA4",
        "source_hashes": {
            name: _sha256(name)
            for name in (
                "caffnet_train_a.json",
                "caffnet_train_b.json",
                "caffnet_train_btight.json",
                "caffnet_train_widths.json",
                "joint_control/summary.json",
            )
        },
        "scenario_a_paper_protocol": {
            "architecture": "3 hidden ReLU layers, width 200",
            "train_points": 50,
            "test_points": 400,
            "epochs": 50_000,
            "learning_rate": 1e-4,
            "seeds": 5,
            "caffnet_test_mse_mean": caff_mse,
            "paper_caffnet_ff_mse_mean": paper_ff_mse,
            "paper_caffnet_ff_mse_std": paper_ff_std,
            "absolute_mse_difference": abs(caff_mse - paper_ff_mse),
            "within_one_reported_paper_std": abs(caff_mse - paper_ff_mse) <= paper_ff_std,
            "caffnet_max_violation": max(row["test_violation_max"] for row in a_caff),
            "all_caffnet_runs_hard_feasible": all(row["test_violation_max"] == 0.0 for row in a_caff),
            "all_soft_nn_runs_have_test_violation": all(row["test_violation_max"] > 0.0 for row in a_nn),
        },
        "scenario_b_paper_dimensions": {
            "n_out": 5,
            "n_inequality": 5,
            "n_equality": 3,
            "train_points": 1000,
            "test_points": 1000,
            "epochs": 10_000,
            "seeds": 5,
            "mean_objectives": {name: _mean(rows, "test_objective") for name, rows in b_methods.items()},
            "caffnet_max_inequality_violation": max(row["ineq_violation_max"] for row in b_methods["CAffNet"]),
            "caffnet_max_equality_violation": max(row["eq_violation_max"] for row in b_methods["CAffNet"]),
            "all_caffnet_runs_hard_feasible_at_1e-12": all(
                row["ineq_violation_max"] <= 1e-12 and row["eq_violation_max"] <= 1e-12
                for row in b_methods["CAffNet"]
            ),
            "qualification": "The paper does not publish its random matrices/seeds; this dimension-matched clean-room instance cannot reproduce Table 3 objective values, and its inequalities are inactive.",
        },
        "joint_nullspace_control": {
            "seeds": 5,
            "joint_objective_gap_mean": joint["aggregate"]["joint_objective_gap"]["mean"],
            "posthoc_objective_gap_mean": joint["aggregate"]["posthoc_objective_gap"]["mean"],
            "fixed_in_loop_objective_gap_mean": joint["aggregate"]["fixed_in_loop_objective_gap"]["mean"],
            "w_ablation_objective_increase_mean": joint["aggregate"]["w_ablation_objective_increase"]["mean"],
            "joint_max_constraint_residual_mean": joint["aggregate"]["joint_max_constraint_residual"]["mean"],
            "theta_initial_hidden_gradient_mean": joint["aggregate"]["theta_initial_hidden_gradient"]["mean"],
            "phi_initial_hidden_gradient_mean": joint["aggregate"]["phi_initial_hidden_gradient"]["mean"],
            "qualification": joint["interpretation_guardrail"],
        },
        "tight_inequality_stress": {
            "h_scale": 0.25,
            "seeds": 5,
            "caffnet_max_inequality_violation": max(row["ineq_violation_max"] for row in tight_methods["CAffNet"]),
            "caffnet_max_equality_violation": max(row["eq_violation_max"] for row in tight_methods["CAffNet"]),
            "orthogonal_min_of_seed_max_inequality_violation": min(row["ineq_violation_max"] for row in tight_methods["OrthProj"]),
            "parallel_min_of_seed_max_inequality_violation": min(row["ineq_violation_max"] for row in tight_methods["ParProj"]),
            "all_caffnet_runs_hard_feasible_at_2e-12": all(
                row["ineq_violation_max"] <= 2e-12 and row["eq_violation_max"] <= 2e-12
                for row in tight_methods["CAffNet"]
            ),
            "all_fixed_projection_runs_violate_inequalities": all(
                row["ineq_violation_max"] > 0.5
                for name in ("OrthProj", "ParProj")
                for row in tight_methods[name]
            ),
            "qualification": "Synthetic h_scale=0.25 stress variant with an LP-certified all-input feasibility guard; activates inequalities absent from the direct clean-room reconstruction and is not the unpublished Table 3 instance.",
        },
        "width_sweep": {
            "widths": [row["width"] for row in sweep],
            "test_mse": [row["test_mse"] for row in sweep],
            "all_runs_hard_feasible": all(row["test_violation_max"] == 0.0 for row in sweep),
            "monotone_mse_decrease": all(
                right["test_mse"] <= left["test_mse"] for left, right in zip(sweep, sweep[1:])
            ),
            "qualification": "Finite and non-monotone; corroborates feasibility only and is not evidence for universal density.",
        },
    }
    return analysis


def main() -> None:
    result = analyze()
    destination = OUT / "training_analysis.json"
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(destination), "scenario_a_hard_feasible": result["scenario_a_paper_protocol"]["all_caffnet_runs_hard_feasible"], "scenario_b_hard_feasible": result["scenario_b_paper_dimensions"]["all_caffnet_runs_hard_feasible_at_1e-12"]}, indent=2))


if __name__ == "__main__":
    main()
