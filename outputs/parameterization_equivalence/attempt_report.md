# CAffNet Claim 2 — Attempt 3

## Target

Existing judged claim:

> The constraint-affine layer enables joint optimization with network parameters, going beyond fixed orthogonal or parallel projections.

This attempt adversarially tests the stronger representational reading of “going beyond fixed orthogonal projection.” It preserves the narrower empirical distinction between end-to-end constrained training and projection applied only after unconstrained training.

## Analytic result

For `N=I-A†A`, Eq. (4) satisfies

```text
P(f,w)=f-A†(Af-b)+Nw=A†b+N(f+w).
```

Setting `g=f+w` gives exactly the end-to-end orthogonal projection

```text
Proj(g)=g-A†(Ag-b)=A†b+Ng.
```

Therefore, for the same input-dependent `A(x)`, `b(x)`, pseudoinverse, and affine-subset selection, the parameterizations have the same feasible output class whenever the comparison class for `g` can represent the sum class `F+W`.

## Command

```bash
python3 repro/src/run_parameterization_equivalence_stdlib.py \
  --output-dir outputs/parameterization_equivalence \
  --seed 20260719 \
  --cases 500
```

## Numerical coverage

- 500 randomized systems
- Output dimensions 2–8
- Constraint cardinalities 1–10
- Ranks 1–8
- 388 cases with redundant rows
- 415 cases with nontrivial null spaces
- 127 deliberately inconsistent systems
- Input-dependent fixed-rank matrices across 81 inputs
- Explicit rank-change construction
- Finite-difference gradient checks
- Direct-vector optimization trajectory checks
- Restricted-function-class capacity counterexample

## Results

| Check | Maximum error/result |
|---|---:|
| Paired `P(f,w)` versus `Proj(f+w)` output error | `1.9984e-15` |
| Consistent-system feasibility residual | `1.4211e-14` |
| Moore-Penrose identity residual | `7.1054e-15` |
| Feasible-target roundtrip error | `9.9920e-16` |
| `f` versus `w` output-gradient difference | `0` |
| Finite-difference gradient error | `1.3786e-9` |
| Scaled direct-vector trajectory difference | `3.8858e-15` |
| Input-dependent paired output error | `4.4409e-16` |
| Minimum negative-control difference when null space is nontrivial | `8.8491e-3` |
| Inconsistent-system paired output error | `1.4710e-15` |
| Minimum inconsistent-system feasibility residual | `2.0482e-1` |

At an input-dependent rank change, both parameterizations remain equal pointwise but share a unit discontinuity in the constructed output, demonstrating that the extra branch does not remove pseudoinverse rank-change behavior.

## Capacity exception

For `A=[1,0]`, `b=0`, a restricted single class `g_a(x)=ax` cannot represent the target null-space function `x^2`; its best MSE is `0.2040133`. A two-branch sum class `f_a(x)+w_b(x)=ax+bx^2` represents it exactly with zero MSE.

This is a real condition under which `w_phi` adds capacity relative to a fixed restricted base architecture. It does not contradict the equivalence with a sufficiently expressive `g` capable of representing `f+w`.

## Judge-ready, verdict-neutral conclusion

The evidence supports a qualification rather than an unqualified “beyond fixed orthogonal projection” claim. Eq. (4)'s learned null-space branch does not enlarge the representable feasible output set beyond end-to-end orthogonal projection of a sufficiently expressive `g=f+w`. This follows algebraically and is corroborated across randomized, rank-deficient, redundant, inconsistent, and input-dependent systems.

The result does not challenge the narrower benefit over **posthoc** projection: training unconstrained and projecting afterward optimizes a different objective and was substantially worse in Attempts 1 and 2. Separate `f_theta` and `w_phi` networks can also change conditioning, implicit bias, and effective capacity when the single-network comparison is restricted.

## Classification

**Full analytic falsification of the strong representational reading, with explicit assumptions and preserved narrower benefits.** This is a substantive third scientific attempt. Whether the public judge awards a `falsified` verdict depends on whether it interprets the original claim representationally or as an optimization/architecture statement.
