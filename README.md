# Repro — CAffNet: Hard Constraint-Affine NNs (20hdQQQrA4)

Clean-room reproduction of *CAffNet* (Zhao, Lee, Jeon, Yong; arXiv [2605.24437](https://arxiv.org/abs/2605.24437)),
for the [ICML 2026 Agent Reproduction Challenge](https://huggingface.co/spaces/ICML-2026-agent-repro/challenge).
OpenReview `20hdQQQrA4`.

The Constraint-Affine layer uses
`P = f − A†(Af−b) + (I−A†A)w`: an affine projection plus a trainable null-space component. This repository now tests both hard constraint adherence and the previously omitted joint-optimization mechanism.

## Results

| Existing judged claim | Evidence status | Headline evidence |
|---|---|---|
| **C1** hard constraints with arbitrary cardinality | Reduced evidence preserved | Rank-deficient and redundant constraint matrices satisfy the affine identity across randomized instances. |
| **C2** joint optimization beyond posthoc projection | **New reduced mechanism reproduction** | Fully trainable `f_theta` and `w_phi` networks receive nonzero hidden-layer gradients, remain feasible to `2.97e-16`, and outperform a-posteriori projection by `1.6747x` in total objective across five seeds. |
| **C3** universal approximation plus constraint adherence | Partial | Constraint adherence verified to `2.7e-13`; universal approximation remains untested. |

The C2 adversarial baseline is important: a fixed orthogonal projection trained end-to-end also reaches the constrained oracle. The experiment therefore supports the paper's posthoc-versus-joint comparison without claiming that `w_phi` is uniquely necessary when the base network is trained through the projection.

## Reproduce

```bash
# Existing constraint-adherence evidence
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install numpy scipy pytest
python repro/src/run_caffnet.py
python -m pytest repro/tests/

# Dependency-free C2 joint-optimization evidence
python3 repro/src/run_joint_control_stdlib.py \
  --output-dir outputs/joint_control \
  --seeds 0,1,2,3,4 \
  --steps 800 \
  --hidden 12 \
  --lr 0.01 \
  --train-points 64 \
  --test-points 501
```

## C2 headline metrics

- Joint oracle objective gap: `1.0652e-4`.
- Posthoc projection objective gap: `4.3045e-1`.
- Fixed projection trained in-loop gap: `1.1732e-4`.
- `w_phi` ablation objective increase: `7.5016e-1`.
- Joint constraint residual: `2.9698e-16`.
- Unconstrained constraint residual: `1.0103`.
- Independent rerun reproduced all aggregate metrics exactly in approximately 14 seconds on an Apple M2 CPU.

## Scope & honest disclosures

- The new C2 experiment is a substantive reduced mechanism reproduction, not the unreleased paper benchmark.
- It uses a synthetic rotating safety boundary, one width-12 hidden layer, and two encoded one-sided inequalities rather than the paper's three width-200 layers and 11/13 inequalities.
- The paper's benchmark matrices, seeds, control environment, and official code are not public.
- C3 universal approximation remains untested; existing constraint-adherence evidence is preserved.

Logbook: https://huggingface.co/spaces/DineshAI/20hdQQQrA4
