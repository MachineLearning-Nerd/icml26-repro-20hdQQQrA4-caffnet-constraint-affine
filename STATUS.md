# STATUS — CAffNet (`20hdQQQrA4`)

**Session:** theorem and full-neural upgrade. **Last updated:** 2026-07-19.
**State:** evidence published; fresh public verdict pending.

## Evidence

- C1 universal minimal-face and Moore-Penrose certificate closes for arbitrary
  finite `m`, every input, dependent/redundant rows, and all ranks permitted by
  Assumption 3.2.
- C2 exact projector certificate proves the independent null-space path and
  joint gradients. The five-seed control reproduces nonzero gradients,
  load-bearing `w_phi`, and machine-precision feasibility; the successful
  end-to-end fixed-orthogonal adversary is preserved as a qualification.
- C3 independent Euclidean-projection proof closes the full UAT plus adherence
  conjunction for every finite `p>=1`; the finite non-monotone width sweep is
  not used to infer universality.
- Paper-spec Scenario A runs five seeds at width 200 for 50,000 epochs: mean MSE
  `0.0029899` lies within one paper-reported standard deviation of `0.0020`,
  all CAffNet violations are zero, and every soft-NN seed violates.
- Dimension-matched Scenario B is feasible below `1e-12` but is not presented
  as a Table 3 objective reproduction because the source matrices are absent.
- The five-seed tightened inequality stress has fixed-projection violations
  above `0.5` while CAffNet remains below `1.2e-14`; it is explicitly marked
  synthetic and LP-guarded, not the unpublished Table 3 instance.
- 23/23 tests pass; proof JSON reports `all_valid=true`; joint rerun completed
  in 13.73 seconds and reproduced every aggregate value.

## Public-score boundary

The current official verdict references Space SHA `b20dab1...` and is
`toy, toy, inconclusive` (2/6). No score increase is claimed until a fresh
verdict references the published theorem-level revision
`bd3a1cc6a779979920e56cc2291789e3c443a1d5`.

The public 34-file reproduction bundle is available in
[`DineshAI/20hdQQQrA4-artifacts`](https://huggingface.co/buckets/DineshAI/20hdQQQrA4-artifacts#reproduction-caffnet/repro-bundle-v2).
It was clean-downloaded, matched the source inventory byte-for-byte, and
passed all 23 tests plus both fail-closed analyzers after download.

No Git commit or push has been made.
