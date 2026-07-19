# Negative controls

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_kn_01", "created_at": "2026-07-17T20:18:00+00:00", "title": "Three violations"}
-->
Three controls that must **violate** `AP=b` (confirming the projection is load-bearing):
- **Plain `f`** (no projection): residual 3.2 — the unconstrained output violates the constraint.
- **Wrong pseudoinverse `Aᵀ`** (instead of `A†`): residual 16.8 — `Aᵀ` is not a valid pseudoinverse,
  so `AAᵀA≠A` and the Penrose identity fails.
- **Infeasible `b`** (rank-deficient `A`, `b` off range(A)): residual 1.79 — the layer then returns the
  least-squares solution (residual = distance of `b` to range(A)); exact satisfaction requires
  feasibility (paper Assumption 3.2).

Each control confirms a distinct load-bearing element (the projection, the correct pseudoinverse,
and the feasibility assumption).
