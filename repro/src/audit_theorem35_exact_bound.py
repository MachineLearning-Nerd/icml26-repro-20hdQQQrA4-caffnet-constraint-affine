#!/usr/bin/env python3
"""Independent audit of the CAffNet Theorem-3.5 exact-bound verifier."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from decimal import Decimal, getcontext
from fractions import Fraction
from pathlib import Path


F = Fraction


def multiply(a, b):
    return [
        [sum((a[i][k] * b[k][j] for k in range(len(b))), F(0)) for j in range(len(b[0]))]
        for i in range(len(a))
    ]


def independent_matrix_certificate() -> dict[str, object]:
    # For A=[[1,2,0],[0,1,1]], P=A^T(AA^T)^-1 A.  The inverse and projector
    # are derived independently here with exact hand-reduced fractions.
    a = [[F(1), F(2), F(0)], [F(0), F(1), F(1)]]
    at = [list(row) for row in zip(*a)]
    gram_inverse = [[F(1, 3), F(-1, 3)], [F(-1, 3), F(5, 6)]]
    p = multiply(multiply(at, gram_inverse), a)
    eye = [[F(int(i == j)) for j in range(3)] for i in range(3)]
    nproj = [[eye[i][j] - p[i][j] for j in range(3)] for i in range(3)]
    one = max(sum(abs(p[i][j]) for i in range(3)) for j in range(3))
    one_n = max(sum(abs(nproj[i][j]) for i in range(3)) for j in range(3))
    return {
        "p_symmetric": p == [list(row) for row in zip(*p)],
        "p_idempotent": multiply(p, p) == p,
        "null_symmetric": nproj == [list(row) for row in zip(*nproj)],
        "null_idempotent": multiply(nproj, nproj) == nproj,
        "orthogonal_complements": multiply(p, nproj) == [[F(0)] * 3 for _ in range(3)],
        "p_one_norm_squared_le_n": one * one <= 3,
        "null_one_norm_squared_le_n": one_n * one_n <= 3,
    }


def independent_constant_certificate() -> dict[str, object]:
    getcontext().prec = 90
    dims = [1, 2, 3, 5, 8, 64, 127, 512]
    errors = []
    oversized_rejected = 0
    for n in dims:
        root = Decimal(n).sqrt()
        coefficient = Decimal(3) + Decimal(3) * root
        k = Decimal(7) / coefficient
        final = ((Decimal(1) + Decimal(3) * root) * k + k) + k
        errors.append(abs(final - Decimal(7)))
        oversized_rejected += int(coefficient * k * Decimal("1.000001") > Decimal(7))
    return {
        "dimensions": dims,
        "maximum_error": str(max(errors)),
        "all_final_budgets_equal_epsilon": all(e <= Decimal("1e-85") for e in errors),
        "all_oversized_K_controls_rejected": oversized_rejected == len(dims),
    }


def main() -> int:
    verifier = Path(__file__).with_name("verify_theorem35_exact_bound.py")
    completed = subprocess.run(
        [sys.executable, str(verifier)], check=True, text=True, capture_output=True
    )
    reported = json.loads(completed.stdout)
    constants = independent_constant_certificate()
    matrices = independent_matrix_certificate()
    checks = {
        "primary_verifier_passed": reported.get("all_checks_passed") is True,
        "primary_bound_is_exact_claim": reported.get("claim_bound")
        == "||P* - f_t||_p < (3 + 3*sqrt(n_out))*K = epsilon",
        "independent_constant_chain": constants["all_final_budgets_equal_epsilon"],
        "independent_oversized_K_control": constants["all_oversized_K_controls_rejected"],
        "independent_projector_certificate": all(matrices.values()),
    }
    result = {
        "verifier_sha256": hashlib.sha256(verifier.read_bytes()).hexdigest(),
        "verifier_stdout_sha256": hashlib.sha256(completed.stdout.encode()).hexdigest(),
        "independent_constant_certificate": constants,
        "independent_matrix_certificate": matrices,
        "checks": checks,
        "audit_passed": all(checks.values()),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["audit_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
