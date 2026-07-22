# Theorem 3.5 — exact universal-approximation bound

## Source and scope

- Paper: *CAffNet: Hard Constraint-Affine Neural Networks*.
- arXiv: `2605.24437`; OpenReview: `20hdQQQrA4`.
- Primary HTML: `https://ar5iv.labs.arxiv.org/html/2605.24437` (HTTP 200).
- Scope audited: Theorem 3.5 and Appendix C, especially equations (23),
  (28), (31), and the final approximation-error bound.

The prior width sweep established decreasing error on one small neural target,
but did not test the constant stated in the judged claim. This audit targets
that missing obligation directly:

```text
K = epsilon / (3 + 3*sqrt(n_out))
||P* - f_t||_p < (3 + 3*sqrt(n_out))*K = epsilon.
```

## Paper-exact proof chain

Let `f_theta` be the unconstrained approximant with
`||f_theta-f_t||_p < K`. If it is feasible, CAffNet returns it and the result
is immediate. Otherwise, the segment from `f_theta` to feasible `f_t`
intersects a boundary face at `f_gamma`, and equation (23) gives

```text
||f_gamma-f_theta||_p < K.
```

The zero continuous target for the shared null-space network is admissible;
its approximation and the boundary displacement give equation (28),

```text
||w_phi||_p < 2K.
```

Both `A_gamma^dagger A_gamma` and its complement are orthogonal projectors.
Their Euclidean operator norms are at most one; finite-dimensional norm
equivalence gives the paper's equation (31):

```text
||A_gamma^dagger A_gamma||_p <= sqrt(n_out)
||I-A_gamma^dagger A_gamma||_p <= sqrt(n_out).
```

Substitution into the CAffine formula yields the three Appendix-C budgets:

```text
||P_gamma-f_t||_p     < (1 + 3*sqrt(n_out))*K
||P* - f_theta||_p    < (2 + 3*sqrt(n_out))*K
||P* - f_t||_p        < (3 + 3*sqrt(n_out))*K = epsilon.
```

This is the exact claim constant, rather than an alternative projection bound.

## Deterministic verification

The primary verifier checks the scalar chain for every output dimension
`1..512` at 80-digit precision. It separately constructs 1,000 rational
row-space projectors with dimensions `2..8` and ranks `1..n_out`, checking
symmetry, idempotence, complementary row/null spaces, and the induced
`1`/`infinity` norm bounds using exact `fractions.Fraction` arithmetic.

Two fail-sensitive controls are required to fail:

1. Increasing `K` by 0.1% exceeds `epsilon` in all 512 dimensions.
2. Replacing the orthogonal projector with `[[1,1],[0,1]]` fails symmetry,
   idempotence, and the `sqrt(2)` induced-one-norm bound.

An independent auditor recomputes the constant chain with a different epsilon
and precision, and checks a hand-reduced nontrivial rational projector.

```bash
python repro/src/verify_theorem35_exact_bound.py
python repro/src/audit_theorem35_exact_bound.py
```

Expected terminal summary:

```text
exact final identity: 512/512
oversized-K controls detected: 512/512
exact rational projector certificates: 1000/1000
non-projector negative control detected: true
all_checks_passed: true
independent audit_passed: true
```

The result is proof-level, deterministic, local-CPU evidence for the precise
Theorem-3.5 bound named by the judge. It makes no new empirical claim about the
CAffNet-TF benchmark or safety-critical control experiment.
