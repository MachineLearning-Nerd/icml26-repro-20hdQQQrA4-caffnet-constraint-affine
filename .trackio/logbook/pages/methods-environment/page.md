# Methods & environment

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_km_01", "created_at": "2026-07-17T20:17:00+00:00", "title": "Setup"}
-->
**CAffine layer (eq. 8).** `P(x) = f(x) − A(x)†(A(x)f(x) − b(x)) + (I − A(x)†A(x))w(x)`.
A(x): k×n_out input-dependent affine constraint; k ≤ n_out (arbitrary cardinality, incl.
rank-deficient/redundant rows). Penrose identity `AA†A=A` ⇒ `A·P = b` exactly.

**Two methods:** (M1) direct residual `‖AP−b‖∞`; (M2) `P − lstsq(A,b) ∈ null(A)`.

**Environment.** Python 3.12, numpy/scipy, pytest. CPU only. 6/6 tests pass. No official code
released; clean-room numpy/scipy (5-line formula).

**Scope.** C3 (constraint adherence) exact. C2 (joint optimization) = training (out of scope);
C3b (UAT) = existence result.
