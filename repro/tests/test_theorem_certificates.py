"""Tests for CAffNet universal proof certificates and executable controls."""
from fractions import Fraction

import numpy as np
import pytest

from theorem_certificates import (
    all_certificates,
    caffnet_output,
    hard_constraint_certificate,
    joint_optimization_certificate,
    projector_control,
    subconstraint_count,
    uat_error_budget,
    universal_approximation_certificate,
)


def box_with_redundant_constraints(n_out: int = 3) -> tuple[np.ndarray, np.ndarray]:
    eye = np.eye(n_out)
    A = np.vstack([eye, -eye, eye[0], eye[0], -eye[1], eye[2], -eye[2], eye[1], -eye[0]])
    b = np.ones(len(A))
    return A, b


def test_c1_dependency_certificate_closes() -> None:
    result = hard_constraint_certificate()
    assert result["valid"]
    assert result["missing_dependencies"] == {}
    assert result["conclusions"] == ["c1-complete"]


def test_c1_arbitrary_cardinality_count_is_exact() -> None:
    assert subconstraint_count(4, 1) == 4
    assert subconstraint_count(13, 5) == 13 + 78 + 286 + 715 + 1287
    assert subconstraint_count(7, 7) == 2**7 - 1


def test_c1_full_selection_handles_more_constraints_than_outputs() -> None:
    A, b = box_with_redundant_constraints()
    assert A.shape == (13, 3)
    f = np.array([4.0, -3.0, 2.0])
    w = np.array([9.0, -7.0, 5.0])
    output, metadata = caffnet_output(A, b, f, w, p=2)
    assert metadata["feasible_candidates"] > 0
    assert np.max(A @ output - b) <= 1e-9


def test_c2_projector_control_has_two_nonzero_gradient_paths() -> None:
    control = projector_control()
    assert control["gradient_to_f"] == (0, 2, 3)
    assert control["gradient_to_w"] == (0, 2, 3)
    assert control["both_parameter_paths_nonzero"]
    assert control["full_null_displacement_recovered"]


def test_c2_dependency_certificate_closes() -> None:
    result = joint_optimization_certificate()
    assert result["valid"]
    assert result["conclusions"] == ["c2-complete"]


@pytest.mark.parametrize("p", [1.0, 2.0, 4.0])
def test_c3_euclidean_projection_candidate_controls_p_nearest_output(p: float) -> None:
    A = np.vstack([np.eye(2), -np.eye(2)])
    b = np.ones(4)
    target = np.array([0.25, -0.4])
    f = np.array([1.4, -1.8])
    output, _ = caffnet_output(A, b, f, np.zeros(2), p=p)
    assert np.max(A @ output - b) <= 1e-9
    assert np.linalg.norm(output - target, ord=p) <= 3 * np.linalg.norm(f - target, ord=p) + 1e-12


def test_c3_exact_error_budget_closes_at_epsilon() -> None:
    result = uat_error_budget(Fraction(7, 5), 11)
    assert result["final_error_bound"] == result["epsilon"]
    assert result["unconstrained_error"] == Fraction(7, 60)


def test_c3_dependency_certificate_closes() -> None:
    result = universal_approximation_certificate()
    assert result["valid"]
    assert result["conclusions"] == ["c3-complete"]
    assert "identically zero" in result["important_strengthening"]


def test_invalid_inputs_fail_closed() -> None:
    with pytest.raises(ValueError):
        subconstraint_count(0, 2)
    with pytest.raises(ValueError):
        uat_error_budget(Fraction(0), 2)


def test_all_three_claims_pass_together() -> None:
    result = all_certificates()
    assert result["all_valid"]
    assert [claim["claim"] for claim in result["claims"]] == ["C1", "C2", "C3"]
