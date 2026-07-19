# Claim 2 — Joint optimization

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_caffnet_c2_joint_01", "created_at": "2026-07-19T06:55:00+00:00", "title": "C2: jointly trained CAffine layer versus fixed projection"}
-->
**Existing judged Claim 2:** “The constraint-affine layer enables joint optimization with network parameters, going beyond fixed orthogonal or parallel projections.”

**Result: joint optimization reproduced on a reduced, fully trainable safety-control task.** Separate fully trainable `f_theta` and `w_phi` tanh networks were optimized through the CAffine layer on input-dependent rotating affine safety faces. Five deterministic seeds were compared with both the paper-aligned a-posteriori orthogonal projection and an adversarial orthogonal projection trained end-to-end.

| Five-seed mean | Result |
| --- | ---: |
| Joint CAffNet oracle objective gap | `1.0652e-4` |
| Posthoc orthogonal objective gap | `4.3045e-1` |
| Fixed orthogonal trained in-loop gap | `1.1732e-4` |
| Posthoc / joint total-objective ratio | `1.6747x` |
| Objective increase after ablating trained `w_phi` | `7.5016e-1` |
| Joint maximum constraint residual | `2.9698e-16` |
| Unconstrained maximum constraint residual | `1.0103` |
| Initial hidden-gradient norm, `theta` | `3.5106` |
| Initial hidden-gradient norm, `phi` | `3.8494` |

The result directly establishes that gradients reach both network parameter blocks, joint training retains hard feasibility to machine precision, the learned null-space component is load-bearing in the joint parameterization, and joint training substantially outperforms applying projection only after unconstrained training.

**Adversarial qualification:** an orthogonal projection trained end-to-end also reached the constrained oracle. Therefore this experiment supports the paper's narrow posthoc-versus-joint comparison, but does not support a stronger assertion that `w_phi` is uniquely necessary when `f_theta` itself is trained through the projection.

**Reproduce:**

```bash
python3 repro/src/run_joint_control_stdlib.py --output-dir outputs/joint_control --seeds 0,1,2,3,4 --steps 800 --hidden 12 --lr 0.01 --train-points 64 --test-points 501
```

The independent rerun reproduced every aggregate metric exactly in approximately 14 seconds on an Apple M2 CPU with Python 3.14.6 and no third-party dependencies.

**Scope qualification:** this is a substantive reduced mechanism reproduction, not the unreleased paper benchmark. It uses one width-12 hidden layer and two encoded one-sided inequalities rather than the paper's three width-200 layers and 11/13 inequalities. The paper's benchmark matrices, seeds, control environment, and official code are not public.

**Public evidence branch:** https://github.com/MachineLearning-Nerd/icml26-repro-20hdQQQrA4-caffnet-constraint-affine/tree/claim2-joint-optimization

Attempt 2 artifacts: `outputs/joint_control/summary.json`, `learning_curves.csv`, `run.log`, `commands.txt`, `verification.json`, and `attempt_report.md`.

## Attempt 3 — analytic falsification of the strong representational reading

For `N = I - A†A`, Eq. (4) simplifies exactly:

```text
P(f,w) = f - A†(Af-b) + Nw
       = A†b + N(f+w).
```

Setting `g=f+w` gives the ordinary end-to-end orthogonal projection:

```text
Proj(g) = g - A†(Ag-b) = A†b + Ng = P(f,w).
```

Therefore, for the same input-dependent `A(x)`, `b(x)`, pseudoinverse, and active affine subset, the trainable-null-space form and an end-to-end orthogonally projected network have exactly the same feasible output class whenever the comparison network can represent the sum class `F+W`.

A deterministic 500-case audit covered output dimensions 2–8, constraint cardinalities 1–10, ranks 1–8, 388 redundant-row cases, 415 nontrivial null spaces, 127 inconsistent systems, 81 input-dependent evaluations, a rank-change construction, finite-difference gradients, optimization trajectories, and a restricted-class counterexample.

| Analytic audit check | Maximum error/result |
| --- | ---: |
| `P(f,w)` versus `Proj(f+w)` | `1.9984e-15` |
| Consistent feasibility residual | `1.4211e-14` |
| Moore-Penrose residual | `7.1054e-15` |
| `f` versus `w` output-gradient difference | `0` |
| Scaled direct-vector trajectory difference | `3.8858e-15` |
| Input-dependent equivalence error | `4.4409e-16` |

**Falsification with qualification:** Eq. (4)'s extra null-space branch does not enlarge the representable feasible output set beyond end-to-end orthogonal projection of a sufficiently expressive `g=f+w`. It can still improve over **posthoc** projection, change optimization conditioning or implicit bias, and add capacity relative to a deliberately restricted fixed-size comparison network.

```bash
python3 repro/src/run_parameterization_equivalence_stdlib.py --output-dir outputs/parameterization_equivalence --seed 20260719 --cases 500
```

Attempt 3 artifacts: `outputs/parameterization_equivalence/proof.md`, `summary.json`, `random_cases.csv`, `verification.json`, `run.log`, `commands.txt`, and `attempt_report.md`.
