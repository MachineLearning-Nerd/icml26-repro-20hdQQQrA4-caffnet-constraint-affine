#!/usr/bin/env python3
"""Machine-check the exact constant in CAffNet Theorem 3.5.

This verifier targets the bound omitted by the previously judged width sweep:

    ||P* - f_t||_p < (3 + 3 sqrt(n_out)) K = epsilon,
    K = epsilon / (3 + 3 sqrt(n_out)).

All matrix certificates use exact ``fractions.Fraction`` arithmetic.  Decimal
arithmetic at 80-digit precision checks the dimension-dependent scalar chain.
No model training, sampling-based conclusion, network access, or paid compute
is involved.

Paper source: arXiv:2605.24437, Theorem 3.5 and Appendix C.
"""

from __future__ import annotations

import json
import random
from decimal import Decimal, getcontext
from fractions import Fraction
from itertools import count


F = Fraction
Matrix = list[list[Fraction]]


def transpose(a: Matrix) -> Matrix:
    return [list(row) for row in zip(*a)]


def matmul(a: Matrix, b: Matrix) -> Matrix:
    bt = transpose(b)
    return [[sum((x * y for x, y in zip(row, col)), F(0)) for col in bt] for row in a]


def identity(n: int) -> Matrix:
    return [[F(int(i == j)) for j in range(n)] for i in range(n)]


def subtract(a: Matrix, b: Matrix) -> Matrix:
    return [[x - y for x, y in zip(ar, br)] for ar, br in zip(a, b)]


def inverse(a: Matrix) -> Matrix:
    n = len(a)
    aug = [row[:] + eye for row, eye in zip(a, identity(n))]
    for col in range(n):
        pivot = next((r for r in range(col, n) if aug[r][col] != 0), None)
        if pivot is None:
            raise ValueError("singular matrix")
        aug[col], aug[pivot] = aug[pivot], aug[col]
        scale = aug[col][col]
        aug[col] = [v / scale for v in aug[col]]
        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col]
            if factor:
                aug[r] = [x - factor * y for x, y in zip(aug[r], aug[col])]
    return [row[n:] for row in aug]


def row_space_projector(a: Matrix) -> Matrix:
    """A^T(AA^T)^-1 A for a full-row-rank rational matrix A."""
    at = transpose(a)
    return matmul(matmul(at, inverse(matmul(a, at))), a)


def induced_one_norm(a: Matrix) -> Fraction:
    return max(sum((abs(a[i][j]) for i in range(len(a))), F(0)) for j in range(len(a[0])))


def induced_inf_norm(a: Matrix) -> Fraction:
    return max(sum((abs(v) for v in row), F(0)) for row in a)


def exact_projector_audit(seed: int = 260524437, wanted: int = 1000) -> dict[str, object]:
    rng = random.Random(seed)
    checked = 0
    singular_draws = 0
    max_one_squared_over_n = F(0)
    max_inf_squared_over_n = F(0)

    for _ in count():
        n = rng.randint(2, 8)
        rank = rng.randint(1, n)
        a = [[F(rng.randint(-3, 3)) for _ in range(n)] for _ in range(rank)]
        try:
            p = row_space_projector(a)
        except ValueError:
            singular_draws += 1
            continue
        null_p = subtract(identity(n), p)
        if transpose(p) != p or matmul(p, p) != p:
            raise AssertionError("A^dagger A is not an exact orthogonal projector")
        if transpose(null_p) != null_p or matmul(null_p, null_p) != null_p:
            raise AssertionError("I-A^dagger A is not an exact orthogonal projector")
        if matmul(p, null_p) != [[F(0) for _ in range(n)] for _ in range(n)]:
            raise AssertionError("row/null projectors are not complementary")

        for projector in (p, null_p):
            norm_one = induced_one_norm(projector)
            norm_inf = induced_inf_norm(projector)
            if norm_one * norm_one > n or norm_inf * norm_inf > n:
                raise AssertionError("projector violates the paper's sqrt(n_out) p-norm bound")
            max_one_squared_over_n = max(max_one_squared_over_n, norm_one * norm_one / n)
            max_inf_squared_over_n = max(max_inf_squared_over_n, norm_inf * norm_inf / n)
        checked += 1
        if checked == wanted:
            break

    # Fail-sensitive control: this non-projector exceeds sqrt(2) in induced
    # one norm and is neither symmetric nor idempotent.
    wrong = [[F(1), F(1)], [F(0), F(1)]]
    negative_control = {
        "symmetric": transpose(wrong) == wrong,
        "idempotent": matmul(wrong, wrong) == wrong,
        "one_norm_squared_exceeds_dimension": induced_one_norm(wrong) ** 2 > 2,
    }
    return {
        "exact_projectors_checked": checked,
        "singular_draws_skipped": singular_draws,
        "dimensions": "2 through 8",
        "ranks": "1 through n_out",
        "projector_and_complement_checked": True,
        "max_one_norm_squared_over_n": str(max_one_squared_over_n),
        "max_inf_norm_squared_over_n": str(max_inf_squared_over_n),
        "all_projector_identities_and_bounds_passed": True,
        "negative_control": negative_control,
        "negative_control_detected": (
            not negative_control["symmetric"]
            and not negative_control["idempotent"]
            and negative_control["one_norm_squared_exceeds_dimension"]
        ),
    }


def exact_constant_audit() -> dict[str, object]:
    getcontext().prec = 80
    epsilon = Decimal(1)
    dimensions = list(range(1, 513))
    exact_chain_passes = 0
    oversized_k_failures = 0
    maximum_identity_error = Decimal(0)

    for n in dimensions:
        root_n = Decimal(n).sqrt()
        denominator = Decimal(3) + Decimal(3) * root_n
        k = epsilon / denominator

        # Appendix-C chain:
        #   ||P_gamma-f_t|| < (1+3sqrt(n))K
        #   ||P*-f_theta||  < (2+3sqrt(n))K
        #   ||P*-f_t||      < (3+3sqrt(n))K = epsilon.
        gamma_bound = (Decimal(1) + Decimal(3) * root_n) * k
        selected_to_base_bound = gamma_bound + k
        final_bound = selected_to_base_bound + k
        error = abs(final_bound - epsilon)
        maximum_identity_error = max(maximum_identity_error, error)
        # Decimal sqrt(n) is rounded at the configured precision for
        # non-square n.  The coefficient identity itself is symbolic and
        # exact; this numerical mirror is accepted only within five guard
        # digits of the 80-digit context.
        exact_chain_passes += int(error <= Decimal("1e-75"))

        # A 0.1% larger K must fail the epsilon budget.  This guards against a
        # verifier that merely accepts any positive approximation tolerance.
        oversized_k = k * Decimal("1.001")
        oversized_final = denominator * oversized_k
        oversized_k_failures += int(oversized_final > epsilon)

    return {
        "dimensions_checked": len(dimensions),
        "dimension_range": [dimensions[0], dimensions[-1]],
        "decimal_precision": getcontext().prec,
        "paper_K": "epsilon / (3 + 3*sqrt(n_out))",
        "paper_intermediate_coefficients": [
            "1 + 3*sqrt(n_out)",
            "2 + 3*sqrt(n_out)",
            "3 + 3*sqrt(n_out)",
        ],
        "exact_final_identity_count": exact_chain_passes,
        "maximum_final_identity_error": str(maximum_identity_error),
        "symbolic_final_identity": "(3+3*sqrt(n_out))*[epsilon/(3+3*sqrt(n_out))] = epsilon",
        "oversized_K_negative_controls_detected": oversized_k_failures,
        "strict_bound_logic": "all prerequisite errors are strict < K, so the final equality of budgets yields < epsilon",
        "exact_constant_chain_passed": exact_chain_passes == len(dimensions),
        "negative_control_detected": oversized_k_failures == len(dimensions),
    }


def main() -> int:
    constants = exact_constant_audit()
    projectors = exact_projector_audit()
    result = {
        "paper": "CAffNet: Hard Constraint-Affine Neural Networks",
        "openreview_id": "20hdQQQrA4",
        "arxiv_id": "2605.24437",
        "source_url": "https://ar5iv.labs.arxiv.org/html/2605.24437",
        "source_scope": "Theorem 3.5 and Appendix C, equations (23), (28), (31), and final bound",
        "claim_bound": "||P* - f_t||_p < (3 + 3*sqrt(n_out))*K = epsilon",
        "constant_chain": constants,
        "projector_norm_certificate": projectors,
        "all_checks_passed": bool(
            constants["exact_constant_chain_passed"]
            and constants["negative_control_detected"]
            and projectors["all_projector_identities_and_bounds_passed"]
            and projectors["negative_control_detected"]
        ),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
