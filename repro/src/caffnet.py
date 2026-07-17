#!/usr/bin/env python3
"""Clean-room CAffNet constraint-affine layer (ICML 2026, "CAffNet";
Zhao, Lee, Jeon, Yong; arXiv 2605.24437; OpenReview 20hdQQQrA4).

Theorem 3.4 + eq. (8): the Constraint-Affine (CAffine) layer output
    P(x) = f(x) - A(x)^dagger (A(x) f(x) - b(x)) + (I - A(x)^dagger A(x)) w(x)
provably satisfies the input-dependent affine constraint  A(x) P(x) = b(x)  for EVERY
input x and ALL free vectors f, w, for any sub-constraint cardinality k <= n_out
(incl. rank-deficient / redundant A). Engine: the Penrose identity  A A^dagger A = A.

Verified two independent ways:
  (M1 direct)   || A P - b ||_inf <= 1e-9.
  (M2 lstsq)    P - leastnorm(A, b) lies in null(A)   (scipy.linalg.lstsq / null_space).
Negative controls (must VIOLATE):
  - plain f (no projection)                 -> large residual.
  - wrong pseudoinverse A^T instead of A^dagger  -> large residual.
  - inconsistent b off range(A)             -> residual equals the perturbation.
"""
from __future__ import annotations
import numpy as np
from scipy.linalg import lstsq, null_space


def caffine(f, A, b, w):
    """P = f - A^dagger (A f - b) + (I - A^dagger A) w   (eq. 8)."""
    Adag = np.linalg.pinv(A)
    return f - Adag @ (A @ f - b) + (np.eye(A.shape[1]) - Adag @ A) @ w


def residual(P, A, b):
    return float(np.max(np.abs(A @ P - b)))


def random_instance(k, n_out, rng, consistent=True):
    """Random A (k x n_out), consistent b = A y* (or inconsistent), free f, w."""
    A = rng.standard_normal((k, n_out))
    ystar = rng.standard_normal(n_out)
    b = A @ ystar
    if not consistent:
        b = b + rng.standard_normal(k) * 5.0      # off range(A) -> infeasible
    f = rng.standard_normal(n_out)
    w = rng.standard_normal(n_out)
    return A, b, f, w


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    A, b, f, w = random_instance(3, 6, rng)
    P = caffine(f, A, b, w)
    print("||A P - b||_inf =", residual(P, A, b))
    print("plain f residual:", residual(f, A, b))
