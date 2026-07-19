# CAffNet Eq. (4): Representational Equivalence Audit

## Statement

Fix an input `x`, an affine constraint matrix `A=A(x)`, right-hand side `b=b(x)`, and the Moore-Penrose pseudoinverse `A†`. Define

```text
N = I - A†A.
```

CAffNet Eq. (4) is

```text
P(f,w) = f - A†(Af-b) + Nw.
```

Expanding gives

```text
P(f,w)
  = f - A†Af + A†b + Nw
  = Nf + A†b + Nw
  = A†b + N(f+w).
```

For `g=f+w`, the ordinary end-to-end orthogonal affine projection is

```text
Proj(g) = g - A†(Ag-b)
        = A†b + Ng.
```

Therefore

```text
P(f,w) = Proj(f+w)
```

pointwise for every `f`, `w`, `A`, and `b` for which the same Moore-Penrose inverse and affine-subset selection are used.

## Feasible output set

When `b` is in the range of `A`:

- `A†b` is a particular solution of `Ay=b`;
- `N=I-A†A` is the orthogonal projector onto `null(A)`;
- the complete feasible affine set is `A†b + null(A)`.

As `f+w` ranges over the full output space, both parameterizations therefore represent exactly the same feasible outputs. Rank deficiency and redundant constraints do not change this conclusion because it uses the Moore-Penrose identities rather than a full-row-rank inverse.

When `b` is not in the range of `A`, the equality-constrained feasible set is empty. Both expressions still remain algebraically identical and produce the same least-squares projection, but neither can guarantee `Ay=b`.

## Input-dependent constraints

The identity is pointwise, so it remains true for `A(x)` and `b(x)`. If the rank of `A(x)` changes, `A†(x)` and the shared projected function can be discontinuous. Both parameterizations inherit the same discontinuity; the additional `w` branch does not remove it algebraically.

For inequality constraints handled through an affine active-subset or sub-constraint selection, the identity holds conditional on both comparisons using the same selected `A_gamma(x)` and `b_gamma(x)`. A different learned selection mechanism is a separate source of model capacity and is not established by Eq. (4) alone.

## Function-class qualification

Let `F` be the function class represented by `f_theta`, `W` the class represented by `w_phi`, and `G` the comparison class represented by a single base network `g_psi`.

The CAffNet output class is

```text
A†b + N(F + W),
```

where `F+W={f+w | f in F, w in W}`. The single projected-network output class is

```text
A†b + NG.
```

They are representationally equivalent when `G` contains `F+W`, for example when `g` is sufficiently expressive or explicitly constructed to represent the sum of the two branches.

The `w_phi` branch can add capacity relative to a deliberately restricted fixed-size `g` whose class is not closed under sums. The included counterexample uses a single linear class `g_a(x)=ax`, which cannot represent `x^2`, while two branches `f_a(x)+w_b(x)=ax+bx^2` can. This is an architecture and parameter-budget effect, not a larger affine feasible set introduced by the projection identity.

## Optimization qualification

Representational equivalence does not imply identical training dynamics.

For a loss `L(P)`, direct output-space derivatives satisfy

```text
dL/df = N^T dL/dP,
dL/dw = N^T dL/dP,
dL/dg = N^T dL/dP.
```

The `f` and `w` output-space gradients are identical. In direct vector parameterization, updating both branches with learning rate `eta` is exactly matched by updating `g=f+w` with learning rate `2*eta`; the numerical audit verifies this trajectory equivalence.

Neural networks introduce different Jacobians `J_f` and `J_w`. Their combined tangent kernel, conditioning, parameter count, initialization, and implicit bias can differ from a chosen single `g` network. Separate branches may therefore improve optimization without enlarging the underlying affine feasible output set.

Training an unconstrained network first and projecting only after training is also not equivalent to end-to-end projected training. The posthoc network optimizes the unconstrained objective, while both CAffNet and an end-to-end fixed projection optimize through `N`. The earlier Attempts 1 and 2 empirically support this narrower posthoc-versus-joint distinction.

## Verdict-neutral conclusion

The stronger interpretation that Eq. (4)'s trainable null-space term creates feasible outputs unavailable to any sufficiently expressive end-to-end orthogonally projected network is not supported: the two forms are algebraically identical after setting `g=f+w`.

The analysis does not refute narrower claims that:

- joint constrained training can outperform projection applied only after unconstrained training;
- the two-network parameterization can change optimization conditioning or implicit bias;
- an added `w_phi` network can increase capacity relative to a fixed restricted comparison architecture;
- learned active-subset selection can introduce behavior outside the fixed-`A,b` equivalence analyzed here.
