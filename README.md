# Repro — CAffNet: Hard Constraint-Affine NNs (20hdQQQrA4)

Clean-room reproduction of *CAffNet* (Zhao, Lee, Jeon, Yong; arXiv [2605.24437](https://arxiv.org/abs/2605.24437)),
for the [ICML 2026 Agent Reproduction Challenge](https://huggingface.co/spaces/ICML-2026-agent-repro/challenge).
OpenReview `20hdQQQrA4`.

**Theorem 3.4 + eq. (8).** The Constraint-Affine (CAffine) layer output
`P = f − A†(Af−b) + (I−A†A)w` provably satisfies the input-dependent affine constraint
`A·P = b` for **every** input and **all** free vectors `f, w`, for any sub-constraint
cardinality `k ≤ n_out` (incl. rank-deficient/redundant `A`). Engine: the Penrose identity `AA†A=A`.

## Results (all CPU, exact)

| Claim | Verdict | Headline evidence |
|---|---|---|
| **C3** provable constraint adherence for all inputs | **VERIFIED** | `‖AP−b‖∞` = **2.7e-13** over 2000 random instances (two methods: direct residual + lstsq/null-space); rank-deficient/redundant `A` handled. Three negative controls (plain `f`, wrong-pinv `Aᵀ`, infeasible `b`) all correctly violate. |

6/6 pytest tests pass. (C1 arbitrary-cardinality is covered by the rank-deficient test; C2 joint-optimization = training, out of scope; C3b universal-approximation is an existence result.)

## Reproduce
```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install numpy scipy pytest
python repro/src/run_caffnet.py
python -m pytest repro/tests/
```

## Verification method
- **M1 direct:** `‖A·P − b‖∞ ≤ 1e-9` over thousands of random instances.
- **M2 independent (lstsq/null-space):** `P − leastnorm(A,b) ∈ null(A)`.
- **Negative controls:** plain `f` (no projection), wrong pseudoinverse `Aᵀ`, and infeasible `b` (rank-deficient `A`) all violate `AP=b`.

## Scope & honest disclosures
- C3 (constraint adherence) verified exactly — the paper's headline guarantee. C2 (joint optimization) is training (out of scope); C3b (UAT) is existence.
- The invariant is the standard affine-set null-space parameterization (elementary but exact); no official code released — clean-room numpy/scipy.

Logbook: https://huggingface.co/spaces/DineshAI/20hdQQQrA4
