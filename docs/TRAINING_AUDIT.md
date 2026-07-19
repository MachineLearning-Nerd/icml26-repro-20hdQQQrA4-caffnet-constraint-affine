# Neural experiment audit

`repro/src/analyze_training.py` validates raw JSON shape, seed coverage,
published hyperparameters, source hashes, aggregate metrics, and interpretation
boundaries. Its generated output is `outputs/training_analysis.json`.

## Scenario A — paper-published protocol

The clean-room reconstruction uses Appendix D.1's target and four piecewise
constraints, three hidden ReLU layers of width 200, 50 uniform training points,
400 linearly spaced test points, 50,000 full-batch epochs, Adam at `1e-4`, and
five seeds.

| Metric | Reproduction | Paper Table 2 |
|---|---:|---:|
| CAffNet-FF test MSE | `0.0029899` | `0.0020±0.0032` |
| Absolute mean difference | `0.0009899` | within one reported SD |
| CAffNet maximum violation | `0.0` | `0.0` |
| Soft-NN seeds with violation | `5/5` | violations reported |

This is a direct neural task at the paper's published FF width, data count,
optimizer, epoch count, and seed count. The code uses the exact 1D equivalent
of Equation (6): nearest feasible candidate is clamping to the tight interval.

## Scenario B — dimension-matched clean room

The solver uses the published dimensions `n_out=n_ineq=5`, `n_eq=3`, 1,000
training and test inputs, 10,000 epochs, and five seeds. CAffNet maximum
inequality/equality residuals are `8.88e-16` and `4.44e-15`. However, the paper
does not publish its random matrices or seeds; the reconstructed instance's
inequalities are inactive. Its objective values cannot be compared with Table
3 and are not used as verification of those numerical results.

## Joint null-space control

The dependency-free control trains separate `f_theta` and `w_phi` networks on
input-dependent rotating affine faces for five seeds.

| Metric | Mean |
|---|---:|
| Joint oracle objective gap | `1.0652e-4` |
| Posthoc orthogonal gap | `4.3045e-1` |
| Fixed orthogonal trained in-loop gap | `1.1732e-4` |
| Objective increase after `w_phi` ablation | `7.5016e-1` |
| Joint maximum constraint residual | `2.9698e-16` |
| Initial hidden gradient, `theta` | `3.5106` |
| Initial hidden gradient, `phi` | `3.8494` |

The result establishes joint trainability, a load-bearing null-space head, and
superiority to posthoc projection on this control. Because the end-to-end fixed
orthogonal adversary also reaches the oracle, it does not establish that
`w_phi` is uniquely necessary or that CAffNet universally beats every
end-to-end projected architecture.

## Tight inequality stress

The direct clean-room Scenario B inherits the paper's published construction
but its sampled inequalities are inactive. A separate `h_scale=0.25` variant
tightens them and adds an exact linear-program feasibility guard over all input
cube corners. This preserves all-input feasibility while forcing the inequality
mechanism to matter.

Across five seeds, the smallest per-seed maximum inequality violation is
`0.8070` for fixed orthogonal and `0.5909` for fixed parallel projection.
CAffNet's worst inequality and equality residuals are `5.77e-15` and
`1.13e-14`. This directly stresses the arbitrary-combination filter beyond the
fixed projections, but is a synthetic adversarial variant—not the paper's
unpublished Table 3 matrices or an objective-value reproduction.

## Width sweep boundary

Widths 25, 50, 100, 200, and 400 all have zero violations, but test MSE is not
monotone. This finite sweep corroborates adherence and is explicitly excluded
from the universal-approximation proof; C3 instead uses the theorem certificate.
