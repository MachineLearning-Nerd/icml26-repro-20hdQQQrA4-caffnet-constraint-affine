# Executive summary

---
<!-- trackio-cell
{"type":"markdown","id":"cell_exec_caff","created_at":"2026-07-19T10:20:41+00:00","title":"Executive summary","pinned":true,"pinned_at":"2026-07-19T10:20:41+00:00"}
-->
The previous evidence established the affine identity on random matrices and a
reduced joint-training mechanism, but left universal approximation untested.
This revision changes the evidence type and scale.

C1 now follows the complete minimal-face existence and Moore-Penrose argument
for every input, arbitrary finite constraint cardinality, and dependent or
redundant rows. C2 has an exact null-space projector and gradient certificate,
backed by a five-seed trained control. C3 has an independent UAT proof: with an
exactly zero null-space network, the Euclidean projection of an unconstrained
approximant is one of CAffNet's enumerated feasible candidates; norm equivalence
then transfers any underlying UAT error to CAffNet.

The paper-spec feedforward Scenario A is also executed at three hidden layers,
width 200, 50/400 train/test points, 50,000 epochs, Adam `1e-4`, and five seeds.
Mean CAffNet MSE is `0.0029899` versus paper `0.0020±0.0032`; all CAffNet
violations are zero, while every soft-NN seed violates.

## Scope and verification

| Item | Result |
| --- | --- |
| Paper identity | SHA-256 `33db8038...6da9520` |
| Universal DAGs | C1 valid; C2 valid; C3 valid; combined `all_valid=true` |
| Full test suite | 23/23 pass on CPU |
| Paper-spec neural task | 5 seeds; MSE within one reported SD; zero violations |
| Joint control | 5 seeds; both parameter paths receive gradients; hard residual `2.97e-16` |
| Tight inequality stress | 5 seeds; fixed baselines violate by `>0.5`; CAffNet stays below `1.2e-14` |
| Evidence boundary | Dimension-matched solver and finite width sweep are qualified, not overstated |

---
<!-- trackio-cell
{"type":"figure","id":"cell_poster_caff","created_at":"2026-07-19T10:20:42+00:00","title":"CAffNet proof and neural audit poster","pinned":true,"pinned_at":"2026-07-19T10:20:42+00:00"}
-->
````html
<!-- poster_embed.html -->
<iframe src="poster_embed.html" title="CAffNet theorem and neural audit poster" width="100%" height="700" loading="lazy"></iframe>
````
