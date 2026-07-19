"""Regression tests for the neural-experiment evidence summary."""
from analyze_training import analyze


def test_paper_spec_scenario_a_is_complete_and_feasible() -> None:
    result = analyze()["scenario_a_paper_protocol"]
    assert result["seeds"] == 5
    assert result["all_caffnet_runs_hard_feasible"]
    assert result["all_soft_nn_runs_have_test_violation"]
    assert result["within_one_reported_paper_std"]


def test_dimension_matched_scenario_b_is_honestly_qualified() -> None:
    result = analyze()["scenario_b_paper_dimensions"]
    assert result["n_out"] == 5 and result["n_inequality"] == 5 and result["n_equality"] == 3
    assert result["all_caffnet_runs_hard_feasible_at_1e-12"]
    assert "cannot reproduce Table 3" in result["qualification"]


def test_joint_control_preserves_adversarial_baseline_result() -> None:
    result = analyze()["joint_nullspace_control"]
    assert result["joint_objective_gap_mean"] < 1e-3
    assert result["w_ablation_objective_increase_mean"] > 0.5
    assert result["theta_initial_hidden_gradient_mean"] > 0
    assert result["phi_initial_hidden_gradient_mean"] > 0
    assert result["fixed_in_loop_objective_gap_mean"] < 1e-3
    assert "may refute" in result["qualification"]


def test_tight_stress_separates_full_caffnet_from_fixed_projections() -> None:
    result = analyze()["tight_inequality_stress"]
    assert result["seeds"] == 5
    assert result["all_caffnet_runs_hard_feasible_at_2e-12"]
    assert result["all_fixed_projection_runs_violate_inequalities"]
    assert result["orthogonal_min_of_seed_max_inequality_violation"] > 0.8
    assert "not the unpublished Table 3 instance" in result["qualification"]


def test_width_sweep_is_not_misrepresented_as_uat_proof() -> None:
    result = analyze()["width_sweep"]
    assert result["all_runs_hard_feasible"]
    assert not result["monotone_mse_decrease"]
    assert "not evidence for universal density" in result["qualification"]
