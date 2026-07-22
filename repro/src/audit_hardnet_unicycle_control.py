#!/usr/bin/env python3
"""Exact-task CAffNet/HardNet-Aff control-layer audit (local CPU only).

The task constants are transcribed from CAffNet arXiv:2605.24437, Section 4.3
and Appendix D.3.  HardNet-Aff is the one-sided specialization of the official
implementation at https://github.com/azizanlab/hardnet (hardnet_aff.py):

    f + lstsq(A, -relu(A f - b)).solution

The unavailable trained network is not reconstructed.  Instead, the audit sets
the neural correction to zero for every method and isolates the enforcement
layers around the paper's exact saturated nominal PID controller.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np


AR5IV_URL = "https://ar5iv.labs.arxiv.org/html/2605.24437"
HARDNET_COMMIT = "4f3ebe496c4081489c486e2711f25697a4c312fa"
HARDNET_SOURCE_SHA256 = "7fb545ba991719d89cca1553bd4aef824a416ea2ad07cf97e54565c405586f1b"
HARDNET_URL = f"https://github.com/azizanlab/hardnet/blob/{HARDNET_COMMIT}/hardnet_aff.py"
DT = 0.1
STEPS = 150
KAPPA = 10.0

AX = np.array(
    [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
    dtype=float,
)
BX = np.array([1, 5, 2, 4, np.pi, np.pi], dtype=float)
AU = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=float)
BU = np.array([1, 0.01, 0.5, 0.5], dtype=float)

OBS = (
    (
        np.array(
            [[0.4472, -0.8944], [0.7071, 0.7071], [-0.2425, 0.9701],
             [-0.7071, -0.7071], [-0.8944, -0.4472]],
            dtype=float,
        ),
        np.array([-0.2184, -0.5303, 0.6219, 1.1667, 1.4368], dtype=float),
    ),
    (
        np.array(
            [[-0.9685, 0.2489], [0.9417, 0.3363], [-0.3714, 0.9285],
             [0.3714, 0.9285], [-0.9417, -0.3363], [-0.2976, -0.9547]],
            dtype=float,
        ),
        np.array([1.2755, -1.7670, -0.8511, -2.0249, 2.3274, 2.6868], dtype=float),
    ),
    (
        np.array(
            [[-0.9191, 0.3939], [0.8944, 0.4472], [0.9703, -0.2419],
             [-0.8701, -0.4930], [0.0000, -1.0000]],
            dtype=float,
        ),
        np.array([2.9916, -1.9975, -2.5305, 2.4854, 0.1000], dtype=float),
    ),
)

KP = np.diag([0.01, 0.2, 0.0])
KI = np.diag([0.05, 0.005, 0.0])
KD = np.diag([0.0, 0.01, 0.0])
PID_MAP = np.array([[1, 0, 0], [0, 1, 1]], dtype=float)


def dynamics_matrix(x: np.ndarray) -> np.ndarray:
    return np.array(
        [[np.cos(x[2]), 0.0], [np.sin(x[2]), 0.0], [0.0, 1.0]],
        dtype=x.dtype,
    )


def obstacle_h_gradient(position: np.ndarray, a: np.ndarray, b: np.ndarray):
    """Paper's buffered smooth union and its position gradient."""
    edges = a.astype(position.dtype) @ position - b.astype(position.dtype)
    z = position.dtype.type(KAPPA) * edges
    shifted = z - np.max(z)
    exp_shifted = np.exp(shifted)
    weights = exp_shifted / np.sum(exp_shifted)
    h = (np.max(z) + np.log(np.sum(exp_shifted)) - np.log(len(b))) / KAPPA
    return float(h), weights @ a.astype(position.dtype)


def constraints(x: np.ndarray):
    """Return the paper's A(x)u<=b(x), plus stable row labels."""
    g = dynamics_matrix(x)
    rows, bounds, labels = [], [], []
    for j, (a_obs, b_obs) in enumerate(OBS, start=1):
        h, grad_position = obstacle_h_gradient(x[:2], a_obs, b_obs)
        grad = np.array([grad_position[0], grad_position[1], 0.0], dtype=x.dtype)
        rows.append(-(grad @ g))
        bounds.append(h)  # f=0 and alpha(h)=h in Section 4.3
        labels.append(f"obstacle_{j}")

    rows.extend(AX.astype(x.dtype) @ g)
    bounds.extend(BX.astype(x.dtype) - AX.astype(x.dtype) @ x)
    labels.extend([f"state_{i}" for i in range(6)])
    rows.extend(AU.astype(x.dtype))
    bounds.extend(BU.astype(x.dtype))
    labels.extend([f"control_{i}" for i in range(4)])
    return np.asarray(rows, dtype=x.dtype), np.asarray(bounds, dtype=x.dtype), labels


def hardnet_aff(raw_u: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Official HardNet-Aff one-sided enforcement formula."""
    violation = np.maximum(a @ raw_u - b, 0)
    return raw_u + np.linalg.lstsq(a, -violation, rcond=None)[0]


def hardnet_violated_rows_only(raw_u: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Causal ablation: remove the satisfied rows that dilute least squares."""
    active = a @ raw_u - b > 0
    if not np.any(active):
        return raw_u.copy()
    return hardnet_aff(raw_u, a[active], b[active])


def caffnet(raw_u: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Equation (8)/(12), w_phi=0 correction, all faces of size 1 and 2."""
    tol = 3e-6 if raw_u.dtype == np.float32 else 2e-9
    if np.max(a @ raw_u - b) <= tol:
        return raw_u.copy()
    candidates = []
    for cardinality in (1, 2):
        for indices in itertools.combinations(range(len(b)), cardinality):
            a_face = a[list(indices)]
            b_face = b[list(indices)]
            candidate = raw_u - np.linalg.pinv(a_face) @ (a_face @ raw_u - b_face)
            if np.max(a @ candidate - b) <= tol:
                candidates.append(candidate)
    if not candidates:
        raise RuntimeError("CAffNet face enumeration found no feasible candidate")
    return min(candidates, key=lambda u: float(np.linalg.norm(u - raw_u)))


def polygon_margin(x: np.ndarray, obstacle_index: int) -> float:
    a_obs, b_obs = OBS[obstacle_index]
    return float(np.max(a_obs @ x[:2] - b_obs))


def nominal_pid(x, integral_error, previous_error, dtype):
    theta = x[2]
    transform = np.array(
        [[np.cos(theta), np.sin(theta), 0], [-np.sin(theta), np.cos(theta), 0], [0, 0, 1]],
        dtype=dtype,
    )
    error = transform @ (-x)  # x_ref=[0,0,0]
    integral_error = integral_error + dtype.type(DT) * error
    derivative = np.zeros(3, dtype=dtype) if previous_error is None else (error - previous_error) / DT
    u = PID_MAP.astype(dtype) @ (
        KP.astype(dtype) @ error + KI.astype(dtype) @ integral_error + KD.astype(dtype) @ derivative
    )
    u = np.clip(u, np.array([-0.01, -0.5], dtype=dtype), np.array([1.0, 0.5], dtype=dtype))
    return u, integral_error, error


def simulate(method: str, x0=None, dtype=np.float64, row_permutation=None, drop_obstacles=()):
    dtype = np.dtype(dtype)
    x = np.array([-4.5, 0.0, 0.5] if x0 is None else x0, dtype=dtype)
    integral_error = np.zeros(3, dtype=dtype)
    previous_error = None
    min_polygon = np.full(3, np.inf)
    min_smooth_h = np.full(3, np.inf)
    first_collision = [None, None, None]
    max_affine_violation = 0.0
    max_control_violation = 0.0

    for step in range(STEPS):
        raw_u, integral_error, previous_error = nominal_pid(
            x, integral_error, previous_error, dtype
        )
        a, b, labels = constraints(x)
        keep = np.array([label not in {f"obstacle_{j}" for j in drop_obstacles} for label in labels])
        a_used, b_used = a[keep], b[keep]
        if row_permutation is not None:
            order = np.asarray(row_permutation)[keep]
            order = np.argsort(order)
            a_used, b_used = a_used[order], b_used[order]

        if method == "nominal":
            u = raw_u
        elif method == "hardnet":
            u = hardnet_aff(raw_u, a_used, b_used)
        elif method == "hardnet_violated_only":
            u = hardnet_violated_rows_only(raw_u, a_used, b_used)
        elif method == "caffnet":
            u = caffnet(raw_u, a_used, b_used)
        else:
            raise ValueError(method)

        # Section 4.3 saturates both nominal and applied unicycle commands.
        u = np.clip(u, np.array([-0.01, -0.5], dtype=dtype), np.array([1.0, 0.5], dtype=dtype))
        max_affine_violation = max(max_affine_violation, float(np.max(a @ u - b)))
        max_control_violation = max(max_control_violation, float(np.max(AU @ u - BU)))

        for obstacle_index, (a_obs, b_obs) in enumerate(OBS):
            margin = polygon_margin(x, obstacle_index)
            min_polygon[obstacle_index] = min(min_polygon[obstacle_index], margin)
            h, _ = obstacle_h_gradient(x[:2], a_obs, b_obs)
            min_smooth_h[obstacle_index] = min(min_smooth_h[obstacle_index], h)
            if margin <= 0 and first_collision[obstacle_index] is None:
                first_collision[obstacle_index] = step

        x = x + dtype.type(DT) * (dynamics_matrix(x) @ u)
        x[2] = (x[2] + dtype.type(np.pi)) % dtype.type(2 * np.pi) - dtype.type(np.pi)

    return {
        "final_state": [float(v) for v in x],
        "final_goal_distance": float(np.linalg.norm(x[:2])),
        "minimum_polygon_margins": [float(v) for v in min_polygon],
        "minimum_smooth_cbf_values": [float(v) for v in min_smooth_h],
        "collisions": [bool(v <= 0) for v in min_polygon],
        "first_collision_steps": first_collision,
        "max_affine_violation": max_affine_violation,
        "max_control_bound_violation": max_control_violation,
    }


def run_audit():
    main = {method: simulate(method) for method in ("nominal", "hardnet", "caffnet")}

    # Attempt 2: exact same calculation in float32 and under five row permutations.
    float32 = {method: simulate(method, dtype=np.float32) for method in ("hardnet", "caffnet")}
    permutation_runs = []
    for seed in range(5):
        permutation = np.random.default_rng(seed).permutation(13)
        permutation_runs.append(
            {method: simulate(method, row_permutation=permutation) for method in ("hardnet", "caffnet")}
        )

    # Attempt 3: a 3x3x3 neighborhood around the paper's exact test state.
    neighborhood = []
    for dx, dy, dtheta in itertools.product((-0.03, 0.0, 0.03), repeat=3):
        x0 = [-4.5 + dx, dy, 0.5 + dtheta]
        neighborhood.append({method: simulate(method, x0=x0) for method in ("hardnet", "caffnet")})

    hard_collision_count = sum(any(run["hardnet"]["collisions"]) for run in neighborhood)
    caff_collision_count = sum(any(run["caffnet"]["collisions"]) for run in neighborhood)

    # Fail-sensitive/causal controls.
    no_obstacle_filter = simulate("caffnet", drop_obstacles=(1, 2, 3))
    violated_only = simulate("hardnet_violated_only")

    checks = {
        "paper_x0_hardnet_collides_o1_o3": main["hardnet"]["collisions"] == [True, False, True],
        "paper_x0_caffnet_collision_free": main["caffnet"]["collisions"] == [False, False, False],
        "paper_x0_caffnet_affine_feasible": main["caffnet"]["max_affine_violation"] <= 5e-8,
        "paper_x0_hardnet_infeasible": main["hardnet"]["max_affine_violation"] > 1e-3,
        "float32_preserves_outcome": (
            float32["hardnet"]["collisions"] == [True, False, True]
            and float32["caffnet"]["collisions"] == [False, False, False]
        ),
        "row_permutations_preserve_outcome": all(
            run["hardnet"]["collisions"] == [True, False, True]
            and run["caffnet"]["collisions"] == [False, False, False]
            for run in permutation_runs
        ),
        "neighborhood_hardnet_collides_all": hard_collision_count == len(neighborhood),
        "neighborhood_caffnet_safe_all": caff_collision_count == 0,
        "dropping_obstacle_rows_causes_collision": any(no_obstacle_filter["collisions"]),
        "violated_rows_ablation_removes_collision": not any(violated_only["collisions"]),
    }

    return {
        "audit": "CAffNet Section 4.3 exact-task named HardNet-Aff layer audit",
        "scope": "zero learned correction; isolates enforcement layer and does not reconstruct trained weights",
        "sources": {
            "caffnet_ar5iv": AR5IV_URL,
            "hardnet_official_code": HARDNET_URL,
            "hardnet_commit": HARDNET_COMMIT,
            "hardnet_aff_py_sha256": HARDNET_SOURCE_SHA256,
            "source_scope_note": (
                "Official HardNet's bundled CBF dataset is a different five-state, "
                "acceleration-controlled task; only its published enforcement layer is used here."
            ),
        },
        "paper_parameters": {
            "initial_state": [-4.5, 0.0, 0.5],
            "goal": [0.0, 0.0],
            "dt": DT,
            "steps": STEPS,
            "kappa": KAPPA,
            "alpha": "alpha(h)=h",
            "obstacle_edge_counts": [len(b) for _, b in OBS],
            "constraint_rows": 13,
        },
        "attempt_1_paper_x0": main,
        "attempt_2_numeric_robustness": {
            "float32": float32,
            "row_permutation_runs": permutation_runs,
        },
        "attempt_3_initial_state_neighborhood": {
            "cases": len(neighborhood),
            "hardnet_collision_count": hard_collision_count,
            "caffnet_collision_count": caff_collision_count,
        },
        "fail_sensitive_controls": {
            "caffnet_without_obstacle_rows": no_obstacle_filter,
            "hardnet_violated_rows_only": violated_only,
        },
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/hardnet_unicycle_results.json")
    args = parser.parse_args()
    result = run_audit()
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(serialized)
    print(json.dumps(result["attempt_1_paper_x0"], indent=2, sort_keys=True))
    print(json.dumps(result["attempt_3_initial_state_neighborhood"], sort_keys=True))
    print(json.dumps(result["checks"], indent=2, sort_keys=True))
    print("RESULTS_SHA256=" + hashlib.sha256(serialized.encode()).hexdigest())
    print("all_checks_passed=" + str(result["all_checks_passed"]).lower())
    return 0 if result["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
