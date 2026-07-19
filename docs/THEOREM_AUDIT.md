# Universal theorem and architecture audit

This audit replaces finite-instance extrapolation with proof-level evidence for
the three judged CAffNet claims. The executable source is
`repro/src/theorem_certificates.py`; the generated artifact is
`repro/outputs/theorem_certificates.json`. The audited arXiv v1 PDF has
SHA-256 `33db803823608bdb76db20732f0a3a20aef32c5f4e22f4c4b148a2b7b6da9520`.

## C1 — arbitrary-cardinality hard constraints

Fix any input. Under Assumption 3.2, the feasible set
`S={y:Ay<=b}` is a nonempty polyhedron in `R^n`. CAffNet enumerates every
nonempty subset of one through `min(m,n)` constraint rows; no relation
`m<=n` is required.

A standard polyhedral result gives a minimal face `F` that is an affine space
defined by at most `n` independent active input constraints and satisfies
`F subset S`. Its row set `gamma*` is therefore enumerated. For arbitrary
network outputs `f,w`, Equation (4) gives

```text
P_gamma = f - A_gamma^dagger(A_gamma f-b_gamma)
            + (I-A_gamma^dagger A_gamma)w.
```

Multiplication by `A_gamma*`, together with `AA^dagger A=A` and
`A(I-A^dagger A)=0`, yields `A_gamma* P_gamma*=b_gamma*`. Thus this candidate
lies in the entire minimal face and is feasible. The filtered candidate set is
nonempty. Equation (6) returns `f` only if it is already feasible and otherwise
selects a feasible candidate, proving all `m` inequalities for every input,
including redundant and linearly dependent systems.

## C2 — joint optimization beyond a fixed projection

On any differentiable region with a fixed selected sub-constraint, Equation
(4) rewrites as

```text
P = A^dagger b + N(f_theta+w_phi),
N = I-A^dagger A.
```

`N` is the orthogonal projector onto `null(A)`. Setting `w_phi=0` is the fixed
orthogonal special case. If the face is rank deficient, every null-space
displacement `z` is reachable because `Nz=z`; a learned null-space head can
choose throughout the affine solution face rather than committing to one
orthogonal point or a fixed parallel direction. This is an independently
parameterized optimization path; it is not claimed to enlarge the final
function class when an orthogonally projected `f_theta` is itself trained
end-to-end. Locally,

```text
dP/df = dP/dw = N,
```

so chain rule updates both network blocks whenever the task gradient has a
nonzero null-space component. Candidate filtering and `argmin` make the whole
architecture piecewise differentiable; selection ties are disclosed rather
than ignored.

The exact projector control gives identical nonzero gradients `(0,2,3)` to
both paths and recovers an arbitrary test null displacement. The five-seed
neural control independently shows nonzero gradients, machine-precision hard
feasibility, and a load-bearing `w_phi`. An orthogonal projection trained
end-to-end also reaches its oracle, so this audit supports trainability and the
strict architectural generalization of fixed `w`, not universal empirical
superiority over every end-to-end projection parameterization.

## C3 — universal approximation and hard adherence

The independent proof uses an exactly zero null-space network, avoiding any
need to approximate a face-specific auxiliary target. Let `f_theta` approximate
a feasible target `f_t`, and let `q` be the Euclidean projection of `f_theta`
onto `S(x)`. Projection optimality puts `f_theta-q` in the cone of active
constraint normals. Conic Caratheodory selects at most `n` active rows spanning
that vector, so for one enumerated `gamma`,

```text
q = f_theta-A_gamma^dagger(A_gamma f_theta-b_gamma).
```

Therefore `q` is a feasible Equation (4) candidate. Since Equation (6) chooses
the `p`-nearest feasible candidate,

```text
||P*-f_theta||_p <= ||q-f_theta||_p.
```

Euclidean projection and finite-dimensional norm equivalence give the
conservative all-`p` bound

```text
||q-f_theta||_p <= n ||f_t-f_theta||_p.
```

Choose the underlying universal approximator within `epsilon/(n+1)` of `f_t`.
Triangle inequality then yields

```text
||P*-f_t||_p < (n+1) epsilon/(n+1) = epsilon.
```

This proves density for every finite `p>=1` under Theorem 3.5's assumptions.
Combining it with C1 proves the conjunctive challenge statement: approximation
power is preserved while every output remains feasible.

## Imported dependencies and evidence boundary

The audit explicitly imports standard results rather than claiming to re-prove
them: existence/structure of minimal polyhedral faces, Moore-Penrose projector
identities, the projection theorem for closed convex sets, polyhedral normal
cones, conic Caratheodory, finite-dimensional norm equivalence, and the
underlying network class's universal approximation theorem.

Finite random matrices, width sweeps, and trained networks are valuable
corroboration but are not used to infer a universal theorem.

## Verification

```bash
python repro/src/run_theorem_audit.py
python -m pytest -q repro/tests
```
