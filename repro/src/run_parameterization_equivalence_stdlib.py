#!/usr/bin/env python3
"""Adversarial analytic/numerical audit of CAffNet Eq. (4).

For fixed A, b and Moore-Penrose pseudoinverse A^dagger,

    P(f,w) = f - A^dagger(Af-b) + (I-A^dagger A)w
           = A^dagger b + (I-A^dagger A)(f+w).

Thus P(f,w) equals end-to-end orthogonal projection of g=f+w. This script
checks the identity across dimensions, cardinalities, ranks, redundant rows,
consistent/inconsistent systems, input-dependent matrices, finite-difference
gradients, and direct-vector optimization trajectories. It also constructs a
restricted-function-class example where a second network increases capacity
because the single-network comparison class is not closed under sums.

No third-party dependencies are used.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import random
import statistics
import sys
import time


def transpose(matrix):
    return [list(column) for column in zip(*matrix)]


def matmul(left, right):
    right_t = transpose(right)
    return [[sum(a * b for a, b in zip(row, column)) for column in right_t] for row in left]


def matvec(matrix, vector):
    return [sum(a * b for a, b in zip(row, vector)) for row in matrix]


def identity(size):
    return [[1.0 if row == column else 0.0 for column in range(size)] for row in range(size)]


def matrix_sub(left, right):
    return [[a - b for a, b in zip(left_row, right_row)] for left_row, right_row in zip(left, right)]


def vector_add(*vectors):
    return [sum(values) for values in zip(*vectors)]


def vector_sub(left, right):
    return [a - b for a, b in zip(left, right)]


def vector_scale(value, vector):
    return [value * item for item in vector]


def dot(left, right):
    return sum(a * b for a, b in zip(left, right))


def norm2(vector):
    return math.sqrt(dot(vector, vector))


def max_abs_vector(vector):
    return max((abs(value) for value in vector), default=0.0)


def max_abs_matrix(matrix):
    return max((abs(value) for row in matrix for value in row), default=0.0)


def inverse(matrix, tolerance=1e-12):
    size = len(matrix)
    augmented = [row[:] + ident for row, ident in zip(matrix, identity(size))]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) <= tolerance:
            raise ValueError("singular matrix")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        augmented[column] = [value / scale for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor:
                augmented[row] = [value - factor * pivot_value for value, pivot_value in zip(augmented[row], augmented[column])]
    return [row[size:] for row in augmented]


def independent_columns(matrix, tolerance=1e-10):
    rows = len(matrix)
    columns = len(matrix[0])
    basis = []
    pivots = []
    for column_index in range(columns):
        original = [matrix[row][column_index] for row in range(rows)]
        residual = original[:]
        for unit in basis:
            residual = vector_sub(residual, vector_scale(dot(residual, unit), unit))
        length = norm2(residual)
        if length > tolerance * max(1.0, norm2(original)):
            basis.append(vector_scale(1.0 / length, residual))
            pivots.append(column_index)
    return pivots


def pseudoinverse(matrix):
    """Moore-Penrose inverse from a full-rank factorization A=C R."""
    rows = len(matrix)
    columns = len(matrix[0])
    pivots = independent_columns(matrix)
    rank = len(pivots)
    if rank == 0:
        return [[0.0 for _ in range(rows)] for _ in range(columns)], 0
    c_matrix = [[matrix[row][column] for column in pivots] for row in range(rows)]
    c_t = transpose(c_matrix)
    c_plus = matmul(inverse(matmul(c_t, c_matrix)), c_t)
    r_matrix = matmul(c_plus, matrix)
    r_t = transpose(r_matrix)
    r_plus = matmul(r_t, inverse(matmul(r_matrix, r_t)))
    return matmul(r_plus, c_plus), rank


def projector(matrix, pinv):
    return matrix_sub(identity(len(matrix[0])), matmul(pinv, matrix))


def caffine(matrix, pinv, bound, free_f, free_w):
    correction = matvec(pinv, vector_sub(matvec(matrix, free_f), bound))
    null = projector(matrix, pinv)
    return vector_add(vector_sub(free_f, correction), matvec(null, free_w))


def orthogonal(matrix, pinv, bound, free_g):
    return vector_sub(free_g, matvec(pinv, vector_sub(matvec(matrix, free_g), bound)))


def generated_matrix(rows, columns, rank, rng, x=0.0):
    # U has identity pivot rows plus redundant input-dependent combinations.
    u = []
    for row in range(rows):
        if row < rank:
            u.append([1.0 if row == column else 0.0 for column in range(rank)])
        else:
            u.append([rng.uniform(-1.0, 1.0) + 0.15 * x * (row + 1) * (column + 1) for column in range(rank)])
    # V has identity pivot columns plus input-dependent extra columns.
    v = [[0.0 for _ in range(columns)] for _ in range(rank)]
    for row in range(rank):
        for column in range(columns):
            if column < rank:
                v[row][column] = 1.0 if row == column else 0.0
            else:
                v[row][column] = rng.uniform(-1.0, 1.0) + 0.1 * math.sin((row + 1) * (column + 1) * (x + 1.0))
    return matmul(u, v)


def moore_penrose_residuals(matrix, pinv):
    a_ap_a = matmul(matmul(matrix, pinv), matrix)
    ap_a_ap = matmul(matmul(pinv, matrix), pinv)
    a_ap = matmul(matrix, pinv)
    ap_a = matmul(pinv, matrix)
    return {
        "A_Ap_A_minus_A": max_abs_matrix(matrix_sub(a_ap_a, matrix)),
        "Ap_A_Ap_minus_Ap": max_abs_matrix(matrix_sub(ap_a_ap, pinv)),
        "A_Ap_symmetry": max_abs_matrix(matrix_sub(a_ap, transpose(a_ap))),
        "Ap_A_symmetry": max_abs_matrix(matrix_sub(ap_a, transpose(ap_a))),
    }


def finite_difference_gradient(matrix, pinv, bound, f_value, w_value, target, branch, epsilon=1e-6):
    null = projector(matrix, pinv)
    output = caffine(matrix, pinv, bound, f_value, w_value)
    analytic = matvec(transpose(null), vector_sub(output, target))
    base = f_value if branch == "f" else w_value
    numeric = []
    for index in range(len(base)):
        plus = base[:]
        minus = base[:]
        plus[index] += epsilon
        minus[index] -= epsilon
        if branch == "f":
            plus_output = caffine(matrix, pinv, bound, plus, w_value)
            minus_output = caffine(matrix, pinv, bound, minus, w_value)
        else:
            plus_output = caffine(matrix, pinv, bound, f_value, plus)
            minus_output = caffine(matrix, pinv, bound, f_value, minus)
        plus_loss = 0.5 * dot(vector_sub(plus_output, target), vector_sub(plus_output, target))
        minus_loss = 0.5 * dot(vector_sub(minus_output, target), vector_sub(minus_output, target))
        numeric.append((plus_loss - minus_loss) / (2.0 * epsilon))
    return max_abs_vector(vector_sub(analytic, numeric)), analytic


def direct_trajectory_equivalence(matrix, pinv, bound, f_value, w_value, target, steps=25, learning_rate=0.03):
    f_state = f_value[:]
    w_state = w_value[:]
    g_state = vector_add(f_value, w_value)
    null = projector(matrix, pinv)
    maximum = 0.0
    for _ in range(steps):
        joint_output = caffine(matrix, pinv, bound, f_state, w_state)
        single_output = orthogonal(matrix, pinv, bound, g_state)
        maximum = max(maximum, max_abs_vector(vector_sub(joint_output, single_output)))
        joint_gradient = matvec(transpose(null), vector_sub(joint_output, target))
        single_gradient = matvec(transpose(null), vector_sub(single_output, target))
        f_state = vector_sub(f_state, vector_scale(learning_rate, joint_gradient))
        w_state = vector_sub(w_state, vector_scale(learning_rate, joint_gradient))
        # Because g=f+w, paired direct-vector gradient descent has twice the step.
        g_state = vector_sub(g_state, vector_scale(2.0 * learning_rate, single_gradient))
    return maximum


def inconsistent_bound(matrix, pinv, consistent_bound, rng):
    candidate = [rng.uniform(-1.0, 1.0) for _ in consistent_bound]
    column_projection = matmul(matrix, pinv)
    orthogonal_component = vector_sub(candidate, matvec(column_projection, candidate))
    length = norm2(orthogonal_component)
    if length < 1e-8:
        return None
    return vector_add(consistent_bound, vector_scale(0.5 / length, orthogonal_component))


def run_random_cases(rng, count):
    rows_out = []
    for case_index in range(count):
        columns = rng.randint(2, 8)
        rows = rng.randint(1, 10)
        rank = rng.randint(1, min(rows, columns))
        matrix = generated_matrix(rows, columns, rank, rng, x=rng.uniform(-1.0, 1.0))
        pinv, measured_rank = pseudoinverse(matrix)
        y_star = [rng.uniform(-1.0, 1.0) for _ in range(columns)]
        bound = matvec(matrix, y_star)
        f_value = [rng.uniform(-1.0, 1.0) for _ in range(columns)]
        w_value = [rng.uniform(-1.0, 1.0) for _ in range(columns)]
        target = [rng.uniform(-1.0, 1.0) for _ in range(columns)]
        paired = caffine(matrix, pinv, bound, f_value, w_value)
        single = orthogonal(matrix, pinv, bound, vector_add(f_value, w_value))
        fixed_f = orthogonal(matrix, pinv, bound, f_value)
        mp = moore_penrose_residuals(matrix, pinv)
        f_grad_error, f_gradient = finite_difference_gradient(matrix, pinv, bound, f_value, w_value, target, "f")
        w_grad_error, w_gradient = finite_difference_gradient(matrix, pinv, bound, f_value, w_value, target, "w")
        trajectory_error = direct_trajectory_equivalence(matrix, pinv, bound, f_value, w_value, target)
        feasible_target = vector_add(matvec(pinv, bound), matvec(projector(matrix, pinv), target))
        target_roundtrip = orthogonal(matrix, pinv, bound, feasible_target)
        row = {
            "case": case_index,
            "rows": rows,
            "columns": columns,
            "requested_rank": rank,
            "measured_rank": measured_rank,
            "redundant_rows": rows - rank,
            "paired_output_error": max_abs_vector(vector_sub(paired, single)),
            "consistent_feasibility_residual": max_abs_vector(vector_sub(matvec(matrix, paired), bound)),
            "fixed_f_difference": max_abs_vector(vector_sub(paired, fixed_f)),
            "representable_target_roundtrip_error": max_abs_vector(vector_sub(feasible_target, target_roundtrip)),
            "f_vs_w_gradient_error": max_abs_vector(vector_sub(f_gradient, w_gradient)),
            "f_finite_difference_error": f_grad_error,
            "w_finite_difference_error": w_grad_error,
            "scaled_trajectory_error": trajectory_error,
            "mp_max_residual": max(mp.values()),
            "inconsistent_tested": False,
            "inconsistent_paired_error": 0.0,
            "inconsistent_residual": 0.0,
        }
        if rows > rank and case_index % 3 == 0:
            bad_bound = inconsistent_bound(matrix, pinv, bound, rng)
            if bad_bound is not None:
                bad_paired = caffine(matrix, pinv, bad_bound, f_value, w_value)
                bad_single = orthogonal(matrix, pinv, bad_bound, vector_add(f_value, w_value))
                row["inconsistent_tested"] = True
                row["inconsistent_paired_error"] = max_abs_vector(vector_sub(bad_paired, bad_single))
                row["inconsistent_residual"] = max_abs_vector(vector_sub(matvec(matrix, bad_paired), bad_bound))
        rows_out.append(row)
    return rows_out


def input_dependent_checks(rng):
    fixed_rank_errors = []
    fixed_rank_residuals = []
    for step in range(81):
        x = -1.0 + 2.0 * step / 80.0
        matrix = generated_matrix(6, 5, 3, rng, x=x)
        pinv, _ = pseudoinverse(matrix)
        y_star = [math.sin((index + 1) * x) for index in range(5)]
        bound = matvec(matrix, y_star)
        f_value = [math.cos((index + 1) * x) for index in range(5)]
        w_value = [math.sin((index + 2) * x) for index in range(5)]
        paired = caffine(matrix, pinv, bound, f_value, w_value)
        single = orthogonal(matrix, pinv, bound, vector_add(f_value, w_value))
        fixed_rank_errors.append(max_abs_vector(vector_sub(paired, single)))
        fixed_rank_residuals.append(max_abs_vector(vector_sub(matvec(matrix, paired), bound)))

    rank_change_outputs = []
    for x in [-1.0, -0.1, -1e-3, 0.0, 1e-3, 0.1, 1.0]:
        matrix = [[1.0, 0.0], [0.0, x]]
        bound = [0.0, x]
        pinv, rank = pseudoinverse(matrix)
        f_value = [0.4, 0.0]
        w_value = [-0.2, 0.0]
        paired = caffine(matrix, pinv, bound, f_value, w_value)
        single = orthogonal(matrix, pinv, bound, vector_add(f_value, w_value))
        rank_change_outputs.append({
            "x": x,
            "rank": rank,
            "paired_output": paired,
            "single_output": single,
            "equivalence_error": max_abs_vector(vector_sub(paired, single)),
        })
    return {
        "fixed_rank_max_equivalence_error": max(fixed_rank_errors),
        "fixed_rank_max_feasibility_residual": max(fixed_rank_residuals),
        "rank_change_outputs": rank_change_outputs,
        "rank_change_jump": abs(rank_change_outputs[2]["paired_output"][1] - rank_change_outputs[3]["paired_output"][1]),
    }


def restricted_capacity_counterexample():
    xs = [-1.0 + 2.0 * index / 200.0 for index in range(201)]
    target = [x * x for x in xs]
    # Restricted single branch g_a(x)=a*x: symmetry makes best a exactly zero.
    denominator = sum(x * x for x in xs)
    best_a = sum(x * value for x, value in zip(xs, target)) / denominator
    single_prediction = [best_a * x for x in xs]
    single_mse = sum((prediction - value) ** 2 for prediction, value in zip(single_prediction, target)) / len(xs)
    # Two branches f_a(x)=a*x and w_b(x)=b*x^2 can use a=0,b=1.
    two_branch_prediction = [x * x for x in xs]
    two_branch_mse = sum((prediction - value) ** 2 for prediction, value in zip(two_branch_prediction, target)) / len(xs)
    return {
        "constraint": "A=[1,0], b=0; null-space output is second coordinate",
        "single_comparison_class": "g_a(x)=a*x",
        "two_branch_sum_class": "f_a(x)+w_b(x)=a*x+b*x^2",
        "best_single_a": best_a,
        "single_class_mse": single_mse,
        "two_branch_parameters": {"a": 0.0, "b": 1.0},
        "two_branch_mse": two_branch_mse,
        "interpretation": "w adds capacity only because the restricted single-network class is not closed under sums; a sufficiently expressive g representing f+w removes this difference.",
    }


def stats(values):
    return {
        "mean": statistics.fmean(values),
        "max": max(values),
        "min": min(values),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/parameterization_equivalence")
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--cases", type=int, default=500)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    started = time.perf_counter()
    random_rows = run_random_cases(random.Random(args.seed), args.cases)
    input_checks = input_dependent_checks(random.Random(args.seed + 1))
    capacity = restricted_capacity_counterexample()
    inconsistent_rows = [row for row in random_rows if row["inconsistent_tested"]]
    nontrivial_nullspace_rows = [row for row in random_rows if row["measured_rank"] < row["columns"]]
    payload = {
        "experiment": "CAffNet Eq. (4) parameterization equivalence adversarial audit",
        "identity": "P(f,w)=A^dagger b+(I-A^dagger A)(f+w)=Proj_A,b(g) for g=f+w",
        "assumptions": [
            "The same pointwise A(x), b(x), Moore-Penrose pseudoinverse, and affine-subset selection are used in both parameterizations.",
            "For exact feasibility, b(x) lies in range(A(x)); otherwise both forms produce the same least-squares affine output but cannot satisfy an empty affine set.",
            "Representational equivalence requires the comparison network class G to contain the sum class F+W, or to be sufficiently expressive to represent g=f+w.",
            "The result is pointwise and remains algebraically true for input-dependent and rank-deficient A(x); rank changes can make the shared pseudoinverse map discontinuous.",
        ],
        "configuration": vars(args),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "dependencies": "Python standard library only",
        },
        "runtime_seconds": time.perf_counter() - started,
        "randomized": {
            "cases": len(random_rows),
            "dimensions": {"rows": [min(row["rows"] for row in random_rows), max(row["rows"] for row in random_rows)], "columns": [min(row["columns"] for row in random_rows), max(row["columns"] for row in random_rows)]},
            "ranks": [min(row["measured_rank"] for row in random_rows), max(row["measured_rank"] for row in random_rows)],
            "redundant_cases": sum(row["redundant_rows"] > 0 for row in random_rows),
            "paired_output_error": stats([row["paired_output_error"] for row in random_rows]),
            "consistent_feasibility_residual": stats([row["consistent_feasibility_residual"] for row in random_rows]),
            "moore_penrose_residual": stats([row["mp_max_residual"] for row in random_rows]),
            "representable_target_roundtrip_error": stats([row["representable_target_roundtrip_error"] for row in random_rows]),
            "f_vs_w_gradient_error": stats([row["f_vs_w_gradient_error"] for row in random_rows]),
            "finite_difference_error": stats([max(row["f_finite_difference_error"], row["w_finite_difference_error"]) for row in random_rows]),
            "scaled_direct_trajectory_error": stats([row["scaled_trajectory_error"] for row in random_rows]),
            "nontrivial_nullspace_cases": len(nontrivial_nullspace_rows),
            "negative_control_fixed_f_difference": stats([row["fixed_f_difference"] for row in nontrivial_nullspace_rows]),
            "inconsistent_cases": len(inconsistent_rows),
            "inconsistent_paired_error": stats([row["inconsistent_paired_error"] for row in inconsistent_rows]) if inconsistent_rows else None,
            "inconsistent_residual": stats([row["inconsistent_residual"] for row in inconsistent_rows]) if inconsistent_rows else None,
        },
        "input_dependent": input_checks,
        "restricted_capacity_counterexample": capacity,
        "conclusion": {
            "strong_representational_reading": "falsified under the stated sum-class assumption: w_phi does not enlarge the feasible output set beyond end-to-end orthogonal projection of g=f+w.",
            "narrow_posthoc_reading": "not falsified: training an unconstrained f and projecting only after training can be inferior because it optimizes a different objective.",
            "optimization_dynamics": "not equivalent in general neural parameterizations: separate f_theta and w_phi Jacobians alter conditioning/implicit bias, although direct-vector gradient descent is exactly matched by the single g parameterization after a factor-two learning-rate rescaling.",
            "capacity_exception": "w_phi can add capacity relative to a fixed restricted base architecture whose function class is not closed under sums; this is an architecture/parameter-budget effect, not a larger affine feasible set created by Eq. (4).",
        },
    }
    summary_path = os.path.join(args.output_dir, "summary.json")
    cases_path = os.path.join(args.output_dir, "random_cases.csv")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    with open(cases_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=random_rows[0].keys())
        writer.writeheader()
        writer.writerows(random_rows)
    print(json.dumps(payload, indent=2))
    print("wrote", summary_path)
    print("wrote", cases_path)


if __name__ == "__main__":
    main()
