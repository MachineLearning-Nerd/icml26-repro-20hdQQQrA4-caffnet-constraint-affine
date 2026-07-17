# Conclusion

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_kexec_01", "created_at": "2026-07-17T20:19:00+00:00", "title": "Executive summary", "pinned": true, "pinned_at": "2026-07-17T20:19:05+00:00"}
-->
**C3 reproduced.** *CAffNet* (Zhao et al.; `20hdQQQrA4`) Theorem 3.4 — the CAffine layer provably
satisfies input-dependent affine constraints for every input — is verified to 2.7e-13 over 2000
random instances by two independent methods, with three negative controls.

- **C3 (constraint adherence) — VERIFIED.** `‖AP−b‖∞` = 2.7e-13 (direct) + lstsq/null-space agreement;
  rank-deficient/redundant A handled (arbitrary cardinality).

6/6 pytest tests pass. CPU only, exact.

## Scope & cost
| | This reproduction | Full replication |
| --- | --- | --- |
| Scope | C3 constraint-adherence theorem (exact) | + C2 joint-optimization training, C3b UAT |
| Hardware | 4 vCPU CPU | GPU (for training claims) |
| Time | < 1 min | — |
| Cost | 0 | — |
| Outcome | C3 VERIFIED | — |

## Honest deviations
- C3 (constraint adherence) verified exactly. C2 (joint optimization) is training (out of scope);
  C3b (universal approximation) is an existence result. C1 arbitrary-cardinality is covered by the
  rank-deficient test.
- The invariant is the standard affine-set null-space parameterization (elementary but exact).
  No official code released; clean-room numpy/scipy.
