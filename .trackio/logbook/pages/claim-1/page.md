# Claim 1 — Arbitrary-cardinality hard constraints

---
<!-- trackio-cell
{"type":"markdown","id":"cell_c1_caff","created_at":"2026-07-19T10:20:43+00:00","title":"C1 universal minimal-face certificate"}
-->
## Scored statement

CAffNet embeds hard satisfaction of input-dependent affine constraints with an
arbitrary finite number `m` of rows, including redundant and linearly dependent
systems, under Assumption 3.2's nonempty continuous feasible-set condition.

## Universal certificate

CAffNet enumerates every row subset of cardinality 1 through
`min(m,n_out)`. Every nonempty polyhedron has a minimal face `F` that is an
affine space defined by at most `n_out` independent active input constraints
and satisfies `F subset S`. The corresponding `gamma*` is therefore enumerated.

For arbitrary network outputs `f,w`, Equation (4) is

```text
P_gamma = f - A_gamma^dagger(A_gamma f-b_gamma)
            + (I-A_gamma^dagger A_gamma)w.
```

The Penrose identities give `A_gamma* P_gamma*=b_gamma*`. Thus this candidate
lies in the entire minimal face and is feasible, so the filtered set `S_P` is
nonempty. Equation (6) returns `f` only when it is feasible and otherwise
selects from `S_P`. Every output therefore satisfies all `m` inequalities for
every input; no assumption `m<=n_out` or full row rank is used.

Executable cardinality controls include all 2,379 subsets for `m=13,n_out=5`
and all 127 nonempty subsets for `m=n_out=7`. A direct 13-row/3-output
dependent system with arbitrary `f,w` also returns a feasible candidate.

## Neural corroboration

In the paper-spec Scenario A, all five width-200 CAffNet runs have exactly zero
test violation, while all five soft-NN runs violate. In the dimension-matched
5-output solver, all five CAffNet runs remain feasible below `1e-12`.

**Result: the universal proof and neural evidence support C1 under Assumption
3.2; finite matrices are corroboration only.**

Artifact: [theorem certificate JSON](https://huggingface.co/buckets/DineshAI/20hdQQQrA4-artifacts#reproduction-caffnet/repro-bundle-v2/repro/outputs/theorem_certificates.json)
