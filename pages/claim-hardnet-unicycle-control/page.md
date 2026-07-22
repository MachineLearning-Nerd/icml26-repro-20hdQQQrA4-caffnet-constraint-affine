# Claim 5 — named HardNet baseline on the paper's unicycle task

## Exact challenge scope

The challenge claim is limited to obstacle avoidance: CAffNet avoids the
obstacles while HardNet and the soft-constrained baseline fail. The last judged
revision already established the CAffNet-versus-soft predicates, but the judge
correctly found that its unconstrained controller was not the named HardNet
baseline. This page addresses only that omitted HardNet predicate.

It does **not** claim to recover the authors' trained weights, training seeds,
or goal-arrival result. Those artifacts are not released. The neural correction
is set identically to zero, isolating the enforcement layers around the exact
saturated nominal PID controller. In this audit CAffNet stops safely rather than
reaching the goal (final goal distance `3.0938 m`).

## Paper and implementation scope

- CAffNet source: ar5iv HTML for arXiv `2605.24437`, Section 4.3 and Appendix
  D.3: `https://ar5iv.labs.arxiv.org/html/2605.24437`.
- HardNet source: official `azizanlab/hardnet` revision
  `4f3ebe496c4081489c486e2711f25697a4c312fa`, file `hardnet_aff.py`
  (`sha256:7fb545ba991719d89cca1553bd4aef824a416ea2ad07cf97e54565c405586f1b`).
- The official repository's bundled CBF example is a different five-state,
  acceleration-controlled unicycle. We therefore use only its exact published
  HardNet-Aff enforcement layer on the CAffNet paper's task.

The audit transcribes the paper's velocity-controlled state
`[p_x,p_y,theta]`, input `[v,omega]`, `dt=0.1 s`, 15-second horizon, test state
`[-4.5,0,0.5]`, PID gains, state/input bounds, all three polygon matrices,
smooth-union `kappa=10`, and `alpha(h)=h`. These produce 13 simultaneous affine
rows: three obstacle CBFs, six state CBFs, and four actuator bounds.

For one-sided `A u <= b`, the official HardNet-Aff layer specializes to

```text
u_H = u_raw + lstsq(A, -ReLU(A u_raw - b)).solution.
```

CAffNet enumerates the paper's one- and two-row faces, retains only candidates
that satisfy all 13 rows, and selects the closest one in 2-norm. Both methods
receive the same zero correction and saturated nominal PID command.

## Exact test-state result

| Method | O1 collision | O2 collision | O3 collision | Max affine violation | Minimum polygon margins (O1/O2/O3) |
|---|---:|---:|---:|---:|---:|
| Nominal control | yes | no | yes | `3.2642` | `-0.2732 / 1.5086 / -0.1252` |
| Named HardNet-Aff | **yes** | no | **yes** | `1.9528` | `-0.2798 / 1.5119 / -0.1287` |
| CAffNet | no | no | no | `1.29e-16` | `1.2797 / 2.1009 / 0.1510` |

HardNet first enters O3 at step 39 and O1 at step 54. This matches exactly the
two obstacles named by Section 4.3. CAffNet remains feasible at numerical
precision and does not enter any polygon.

## Three attempts and fail-sensitive controls

1. The paper's exact test state produces HardNet collisions with O1 and O3 and
   zero CAffNet collisions.
2. Repeating in float32 and under five fixed permutations of all constraint
   rows preserves the outcome.
3. Across the `3 x 3 x 3` neighborhood formed by perturbing each initial-state
   coordinate by `{-0.03,0,0.03}`, HardNet collides in `27/27` cases and CAffNet
   in `0/27`.

The result is fail-sensitive in both directions. Removing all obstacle rows
from CAffNet makes it collide. Restricting the HardNet least-squares correction
to only the currently violated rows removes its collision, identifying the
over-cardinal simultaneous row system—not actuator saturation—as the failure
mechanism.

An independent verifier recomputes the committed results and evaluates the
official equation with PyTorch on 256 deterministic systems. Its maximum
PyTorch/NumPy discrepancy is `1.33e-15`.

```bash
python repro/src/audit_hardnet_unicycle_control.py
python repro/src/verify_hardnet_unicycle_control.py
```

Expected summaries:

```text
paper_x0_hardnet_collides_o1_o3: true
paper_x0_caffnet_collision_free: true
neighborhood_hardnet_collides_all: true
neighborhood_caffnet_safe_all: true
all_checks_passed=true
audit_passed: true
```

This additive audit supplies the exact named baseline that the prior evidence
omitted. It leaves the already judged Theorem 3.5, trainable-nullspace, and
arbitrary-cardinality evidence unchanged.
