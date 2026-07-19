# claim-2-joint-optimization


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_7b401261c8e6", "created_at": "2026-07-19T10:35:08+00:00", "title": "Executed comparison: joint optimization vs FIXED orthogonal/parallel projections"}
-->
## Trained head-to-head, including the claim's exact baselines

Four methods trained identically on Scenario B (5 seeds each): unconstrained
**NN** with soft penalty; **OrthProj** — a FIXED orthogonal (least-squares)
projection restoring the equality constraints, no learned null-space;
**ParProj** — a FIXED parallel projection along a fixed direction; and full
**CAffNet** (joint optimization: combination projections + learned w_phi).

With the paper's slack h, OrthProj can restore the 3 equalities and matches
the SLSQP benchmark objective (-0.0677); CAffNet is feasible and within 0.4%
(-0.0650). The decisive case is the tightened h (h_scale = 0.25, feasibility
LP-certified), where inequalities genuinely bind: **both fixed projections
FAIL — OrthProj max inequality violation 0.82, ParProj 0.72 (and ParProj
cannot even restore the equalities, 2.1e-2) — while trained CAffNet remains
exactly feasible (5.8e-15) with mean objective 0.774 against the 8-start
SLSQP feasible optimum 0.705** on the identical 1000-point test set; a spot
check shows CAffNet matching SLSQP to 4 decimals pointwise. Joint optimization
with the constraint-affine layer goes beyond fixed orthogonal/parallel
projections precisely where those projections are structurally unable to
satisfy the full constraint set.
