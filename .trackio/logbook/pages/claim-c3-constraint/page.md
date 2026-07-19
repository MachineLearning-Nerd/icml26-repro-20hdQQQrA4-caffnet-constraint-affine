# Claim 3 — Provable constraint adherence

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_kc3_01", "created_at": "2026-07-17T20:16:00+00:00", "title": "Claim & method"}
-->
**Claim (verbatim):** "CAffNet preserves universal approximation properties while providing provable
guarantees on constraint adherence for all inputs." (Theorem 3.4 + eq. 8)

The CAffine layer output `P = f − A†(Af−b) + (I−A†A)w` is a particular solution of `Ay=b` plus an
arbitrary null-space vector — the standard parameterization of `{y : Ay=b}`. By the Penrose identity
`AA†A=A`, `A·P = b` **exactly**, for every input and all free `f, w`, any cardinality `k≤n_out`.

---
<!-- trackio-cell
{"type": "code", "id": "cell_kc3_02", "created_at": "2026-07-17T20:16:10+00:00", "title": "Verifier", "command": ["python", "repro/src/run_caffnet.py"], "exit_code": 0}
-->
````bash
$ python repro/src/run_caffnet.py
````
- **M1 (direct ‖AP−b‖∞):** max = **2.7e-13** over 2000 random instances (cardinalities k=1..7). ✓
- **M2 (lstsq/null-space):** `P − leastnorm(A,b) ∈ null(A)` on all instances. ✓
- **rank-deficient/redundant A:** 500 instances, all `‖AP−b‖<1e-7`. ✓ (arbitrary cardinality)
- **negative controls:** plain `f` (3.2), wrong-pinv `Aᵀ` (16.8), infeasible `b` (1.79) — all violate. ✓

**=> C3 VERIFIED.** Evidence: `outputs/caffnet_summary.json`.
