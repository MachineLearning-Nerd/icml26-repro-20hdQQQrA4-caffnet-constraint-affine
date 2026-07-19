# Claim 3 — Universal approximation and adherence

---
<!-- trackio-cell
{"type":"markdown","id":"cell_c3_caff","created_at":"2026-07-19T10:20:45+00:00","title":"C3 independent UAT plus hard-feasibility proof"}
-->
## Scored statement

CAffNet preserves the underlying network class's universal approximation
property while guaranteeing constraint adherence for every input.

## Independent UAT proof

Choose `w_phi(x)=0` exactly. For an unconstrained approximant `f_theta`, let
`q` be its Euclidean projection onto the nonempty polyhedron `S(x)`. Projection
optimality places `f_theta-q` in the cone of active normals. Conic Caratheodory
selects at most `n_out` active rows spanning it, so one enumerated subset
`gamma` satisfies

```text
q = f_theta-A_gamma^dagger(A_gamma f_theta-b_gamma).
```

Hence `q` is a feasible Equation (4) candidate. Since Equation (6) chooses the
`p`-nearest feasible candidate,

```text
||P*-f_theta||_p <= ||q-f_theta||_p.
```

Euclidean projection and finite-dimensional norm equivalence give the
conservative bound

```text
||P*-f_target||_p
 <= (n_out+1)||f_theta-f_target||_p.
```

For any `epsilon>0`, underlying density supplies `f_theta` within
`epsilon/(n_out+1)`, so CAffNet error is below `epsilon` for every finite
`p>=1`. C1 simultaneously guarantees that `P*` is feasible for every input.

This proof strengthens the audit boundary by using an exactly zero auxiliary
network; it does not require approximating a changing face-specific null-space
target. Imported projection, normal-cone, Caratheodory, norm-equivalence, and
underlying UAT results are disclosed in the artifact.

The five-width neural sweep is non-monotone and is retained only as zero-
violation corroboration—not as evidence for density.

**Result: both halves of the conjunctive claim are supported at theorem level
under Theorems 3.4–3.5 assumptions.**

Artifact: [self-contained theorem audit](https://huggingface.co/buckets/DineshAI/20hdQQQrA4-artifacts#reproduction-caffnet/repro-bundle-v2/docs/THEOREM_AUDIT.md)
