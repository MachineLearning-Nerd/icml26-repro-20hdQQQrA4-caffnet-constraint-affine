# CAffNet: Hard Constraint-Affine Neural Networks — reproduction

Clean-room CPU reproduction of Zhao, Lee, Jeon, and Yong (2026),
[arXiv:2605.24437](https://arxiv.org/abs/2605.24437), OpenReview
[`20hdQQQrA4`](https://openreview.net/forum?id=20hdQQQrA4).

The primary evidence for the two universal claims is an executable theorem
audit. Neural experiments at the paper's published width/epoch/data scale and
an adversarial joint-optimization control provide independent corroboration.

## Judged claims

| Claim | Primary evidence | Result |
|---|---|---|
| C1 — hard input-dependent affine constraints of arbitrary cardinality | Lemma 3.3 / Theorem 3.4 minimal-face DAG and Moore-Penrose algebra | Universal certificate closes |
| C2 — joint optimization beyond fixed orthogonal/parallel projections | Exact null-space projector/gradient certificate plus five-seed trained control | Architectural claim reproduced with qualification |
| C3 — universal approximation plus hard adherence for every input | Independent Euclidean-projection UAT proof combined with C1 | Full conjunctive certificate closes |

## Proof-level evidence

For C1, CAffNet enumerates every constraint subset of size at most `n_out`.
Every nonempty polyhedron has a minimal face defined by at most `n_out`
independent active rows. For that enumerated subset,

```text
P = f - A^dagger(Af-b) + (I-A^dagger A)w
```

lies in the entire minimal face for arbitrary `f,w`, making the filtered
candidate set nonempty. Equation (6) therefore returns a feasible output for
every input and any finite number `m` of constraints, including dependent and
redundant rows.

For C2, Equation (4) rewrites as `P=A^dagger b+N(f_theta+w_phi)` with
`N=I-A^dagger A`, the projector onto `null(A)`. Locally
`dP/df=dP/dw=N`, so both network blocks receive gradients. The learned head
adds an independently parameterized null-space path and contains fixed
orthogonal projection (`w=0`) as a special case. It is not claimed to enlarge
the final function class when a projected `f_theta` is itself trained
end-to-end.

For C3, choose `w_phi=0` exactly and let `q` be the Euclidean projection of an
unconstrained approximant `f_theta` onto the safe polyhedron. Projection
optimality and conic Caratheodory show that `q` is one of CAffNet's enumerated
feasible candidates. The `p`-nearest selection and norm equivalence give

```text
||P*-f_target||_p <= (n_out+1)||f_theta-f_target||_p.
```

Choosing the underlying approximant within `epsilon/(n_out+1)` proves density
for every finite `p>=1`. C1 supplies hard adherence simultaneously.

## Executed neural evidence

- Paper-spec Scenario A: three hidden ReLU layers of width 200, 50 training
  points, 400 test points, 50,000 epochs, Adam `1e-4`, five seeds. CAffNet mean
  test MSE is `0.0029899` versus paper CAffNet-FF `0.0020±0.0032`; all CAffNet
  violations are exactly zero, while every soft-NN seed violates.
- Dimension-matched Scenario B: `n_out=n_ineq=5`, `n_eq=3`, 1,000 train/test
  inputs, 10,000 epochs, five seeds. All CAffNet runs are feasible below
  `1e-12`. The paper's random matrices/seeds are unavailable and the clean-room
  inequalities are inactive, so Table 3 objectives are not claimed reproduced.
- Joint null-space control: five seeds, separate trainable `f_theta,w_phi`.
  Mean joint oracle gap `1.0652e-4`, mean posthoc gap `4.3045e-1`, mean
  `w_phi` ablation increase `7.5016e-1`, hard residual `2.97e-16`, and nonzero
  gradients to both blocks. A fixed orthogonal projection trained end-to-end
  also reaches its oracle (`1.1732e-4` gap), ruling out a stronger uniqueness
  claim.
- Tight inequality stress: at the paper's 5-output/5-inequality/3-equality
  dimensions, an LP-certified `h_scale=0.25` variant activates the inequalities.
  Across five seeds, fixed orthogonal and parallel projections have maximum
  inequality violations above `0.5`, while CAffNet stays below `1.2e-14` on
  all constraints. This is a disclosed synthetic stress case, not Table 3.

## Repository map

- `repro/src/theorem_certificates.py` — universal C1/C3 proofs and exact C2
  projector/gradient certificate.
- `repro/src/run_theorem_audit.py` — portable certificate renderer.
- `docs/THEOREM_AUDIT.md` — self-contained proof and imported dependencies.
- `repro/src/caffnet_train.py` — paper-spec and dimension-matched PyTorch runs.
- `repro/src/run_joint_control_stdlib.py` — dependency-free five-seed joint
  null-space control.
- `repro/src/analyze_training.py` and `docs/TRAINING_AUDIT.md` — fail-closed
  evidence aggregation and limitations.
- `repro/tests/` — 23 proof, algebra, neural-output, and negative-control tests.

## Reproduce

```bash
uv venv --python 3.12 .venv
uv sync --python .venv/bin/python
.venv/bin/python repro/src/run_theorem_audit.py
.venv/bin/python repro/src/analyze_training.py
.venv/bin/python -m pytest -q repro/tests
.venv/bin/python repro/src/run_caffnet.py
```

The joint neural control reruns in about 14 seconds on an Apple M2 CPU:

```bash
.venv/bin/python repro/src/run_joint_control_stdlib.py \
  --output-dir outputs/joint_control --seeds 0,1,2,3,4 \
  --steps 800 --hidden 12 --lr 0.01 --train-points 64 --test-points 501
```

The longer PyTorch commands and raw logs are recorded under `outputs/`.

## Verified state and scope

- Paper PDF SHA-256:
  `33db803823608bdb76db20732f0a3a20aef32c5f4e22f4c4b148a2b7b6da9520`.
- All three proof DAGs report `valid=true`; combined `all_valid=true`.
- 23/23 tests pass; pinned dependencies are NumPy 2.5.1, SciPy 1.18.0,
  PyTorch 2.13.0, and pytest 9.1.1.
- Imported standard results are disclosed in the theorem audit. Finite random
  matrices and width sweeps are corroboration only, never universal evidence.
