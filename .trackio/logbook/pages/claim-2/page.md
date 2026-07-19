# Claim 2 — Joint null-space optimization

---
<!-- trackio-cell
{"type":"markdown","id":"cell_c2_caff","created_at":"2026-07-19T10:20:44+00:00","title":"C2 projector, gradients, and neural control"}
-->
## Scored statement

The CAffine layer enables joint optimization with network parameters and goes
beyond committing to a fixed orthogonal point or fixed parallel direction.

## Exact architecture certificate

For a locally selected sub-constraint, Equation (4) rewrites as

```text
P = A^dagger b + N(f_theta+w_phi),
N = I-A^dagger A.
```

`N` is the projector onto `null(A)`. Setting `w_phi=0` recovers fixed
orthogonal projection. At a rank-deficient face every null-space displacement
`z` is reachable because `Nz=z`, giving an independently parameterized choice
across the affine face. Away from candidate-selection ties,

```text
dP/df = dP/dw = N.
```

The exact control uses `N=diag(0,1,1)`, recovers the arbitrary null displacement
`(0,5,-7)`, and sends the nonzero gradient `(0,2,3)` into both parameter paths.

## Five-seed executed control

| Metric | Mean |
| --- | ---: |
| Joint oracle objective gap | `1.0652e-4` |
| Posthoc orthogonal gap | `4.3045e-1` |
| Fixed orthogonal trained in-loop gap | `1.1732e-4` |
| Objective increase after `w_phi` ablation | `7.5016e-1` |
| Joint maximum constraint residual | `2.9698e-16` |
| Initial hidden gradient, `theta / phi` | `3.5106 / 3.8494` |

The experiment proves both blocks train jointly, the learned null-space path is
load-bearing in this parameterization, and posthoc projection is worse. The
adversarial end-to-end orthogonal baseline also reaches the oracle, so this
page does not claim `w_phi` is uniquely necessary or universally superior to
every train-through-projection architecture. Selection is piecewise
differentiable, with tie surfaces explicitly outside the local derivative.

An additional five-seed stress test uses the paper's 5-output,
5-inequality, 3-equality dimensions and an LP feasibility guard with
`h_scale=0.25` so the inequalities are active. Every fixed orthogonal and
parallel run has maximum inequality violation above `0.5`; every CAffNet run
stays below `1.2e-14` across inequalities and equalities. This is a disclosed
synthetic stress case, not a claimed reproduction of Table 3 because the
paper's sampled matrices and seeds are unavailable.

**Result: the architectural joint-optimization claim is supported with the
stronger uniqueness interpretation explicitly rejected.**

Artifact: [neural training audit](https://huggingface.co/buckets/DineshAI/20hdQQQrA4-artifacts#reproduction-caffnet/repro-bundle-v2/docs/TRAINING_AUDIT.md)
