# Conclusion

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_kexec_01", "created_at": "2026-07-17T20:19:00+00:00", "title": "Executive summary", "pinned": true, "pinned_at": "2026-07-17T20:19:05+00:00"}
-->
**CAffNet constraint adherence and joint optimization tested.** Existing randomized linear-algebra evidence is preserved, and the previously omitted joint-training claim is now exercised with fully trainable `f_theta` and `w_phi` networks on an input-dependent safety-control task.

- Joint oracle objective gap: `1.0652e-4` across five seeds.
- Posthoc orthogonal projection gap: `4.3045e-1`.
- Joint hard-constraint residual: `2.9698e-16`.
- Ablating trained `w_phi` increases objective by `7.5016e-1`.
- Both hidden parameter blocks receive nonzero gradients.
- Independent rerun reproduced every aggregate metric exactly.

**Verdict-neutral interpretation:** the evidence directly supports joint optimization and superiority over applying projection only after unconstrained training. A fixed orthogonal projection trained end-to-end also reaches the oracle, so the result does not claim that the null-space network is uniquely necessary.

## Scope & cost
| | This reproduction | Full replication |
| --- | --- | --- |
| Scope | Constraint identity plus reduced fully trainable C2 mechanism test | Unreleased learned-optimizer and robot-control benchmarks; universal approximation experiment |
| Hardware | Apple M2 CPU | Paper used V100 for experiments |
| Time | C2 ≈14 s; constraint checks <1 min | — |
| Cost | 0 | — |

## Honest deviations
- The C2 task is synthetic and uses one width-12 hidden layer and two encoded inequalities rather than three width-200 layers and 11/13 inequalities.
- The paper's benchmark matrices, seeds, control environment, and official code are not public.
- Constraint adherence is preserved; universal approximation remains untested.

**Rerun:** `python3 repro/src/run_joint_control_stdlib.py --output-dir outputs/joint_control --seeds 0,1,2,3,4 --steps 800 --hidden 12 --lr 0.01 --train-points 64 --test-points 501`.

**Evidence:** https://github.com/MachineLearning-Nerd/icml26-repro-20hdQQQrA4-caffnet-constraint-affine/tree/claim2-joint-optimization
