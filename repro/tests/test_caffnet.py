#!/usr/bin/env python3
"""Exact tests for CAffNet constraint-affine theorem (20hdQQQrA4, Thm 3.4 + eq. 8)."""
import os, sys
import numpy as np
from scipy.linalg import lstsq
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import caffnet as cn


def test_constraint_adherence_all_cardinalities():
    """A P = b for every input/free-vector/cardinality (M1 direct)."""
    rng = np.random.default_rng(0)
    for _ in range(500):
        n_out = int(rng.integers(2, 8)); k = int(rng.integers(1, n_out + 1))
        A, b, f, w = cn.random_instance(k, n_out, rng)
        P = cn.caffine(f, A, b, w)
        assert cn.residual(P, A, b) < 1e-9


def test_two_method_lstsq_nullspace():
    """M2: P - leastnorm(A,b) lies in null(A)."""
    rng = np.random.default_rng(1)
    for _ in range(200):
        n_out = int(rng.integers(3, 8)); k = int(rng.integers(1, n_out))
        A, b, f, w = cn.random_instance(k, n_out, rng)
        P = cn.caffine(f, A, b, w)
        diff = P - lstsq(A, b)[0]
        Adag = np.linalg.pinv(A)
        assert np.max(np.abs(diff - (np.eye(n_out) - Adag @ A) @ diff)) < 1e-9


def test_rank_deficient_redundant_A():
    """Arbitrary cardinality incl. rank-deficient/redundant rows."""
    rng = np.random.default_rng(2)
    for _ in range(200):
        n_out = int(rng.integers(3, 8))
        A = rng.standard_normal((2, n_out)); A = np.vstack([A, A[0:1]])
        b = A @ rng.standard_normal(n_out)
        f = rng.standard_normal(n_out); w = rng.standard_normal(n_out)
        assert cn.residual(cn.caffine(f, A, b, w), A, b) < 1e-7


def test_negcontrol_plain_f_violates():
    rng = np.random.default_rng(3); A, b, f, w = cn.random_instance(3, 6, rng)
    assert cn.residual(f, A, b) > 0.1


def test_negcontrol_wrong_pseudoinverse_violates():
    rng = np.random.default_rng(4); A, b, f, w = cn.random_instance(3, 6, rng)
    P_wrong = f - A.T @ (A @ f - b) + (np.eye(6) - A.T @ A) @ w
    assert cn.residual(P_wrong, A, b) > 0.1


def test_negcontrol_infeasible_b_least_squares():
    """Rank-deficient A + infeasible b -> least-squares fallback (residual = distance > 0)."""
    rng = np.random.default_rng(5); n_out = 6
    A = np.vstack([rng.standard_normal((2, n_out)), rng.standard_normal((1, n_out))])
    A = np.vstack([A, A[0:1]])  # rank-deficient
    b = A @ rng.standard_normal(n_out) + rng.standard_normal(A.shape[0]) * 5.0
    f = rng.standard_normal(n_out); w = rng.standard_normal(n_out)
    assert cn.residual(cn.caffine(f, A, b, w), A, b) > 0.1


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
