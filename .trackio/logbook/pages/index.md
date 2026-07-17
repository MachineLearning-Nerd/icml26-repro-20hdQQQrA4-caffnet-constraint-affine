# Repro - CAffNet Constraint-Affine NN (20hdQQQrA4)

Clean-room reproduction of *CAffNet* (Zhao et al.; arXiv 2605.24437), OpenReview `20hdQQQrA4`.
Theorem 3.4 + eq. (8): the CAffine layer output provably satisfies input-dependent affine
constraints for every input.

## Claims
| Claim | Statement | Verdict |
| --- | --- | --- |
| **C3** | Provable constraint adherence for all inputs | **VERIFIED** |

(C1 arbitrary-cardinality covered by the rank-deficient test; C2 joint-optimization = training, out of scope; C3b UAT = existence.)

## Pages
- [Claim 3 — constraint adherence](claim-c3-constraint) · [Methods & environment](methods-environment)
- [Negative controls](negative-controls) · [Conclusion](conclusion)

Exact linear algebra, two methods + three negative controls. CPU only.
