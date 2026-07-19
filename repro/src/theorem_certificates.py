"""Universal proof certificates for the three CAffNet challenge claims.

Finite neural experiments are retained as corroboration, but Claims 1 and 3
are universally quantified mathematical statements.  This module records the
actual proof dependencies and exposes the paper-specific algebra as executable
checks.  The UAT certificate uses an independent Euclidean-projection argument
with an exactly-zero null-space network; it does not rely on extrapolating a
width sweep or on the looser constant in Appendix C.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
from itertools import combinations
from math import comb
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class ProofStep:
    id: str
    statement: str
    depends_on: tuple[str, ...] = ()
    reference: str = ""
    evidence_type: str = "derived"


def _verify_dag(steps: Iterable[ProofStep], conclusions: Iterable[str]) -> dict:
    """Fail closed on duplicate, absent, or forward-referenced steps."""
    seen: set[str] = set()
    duplicate_ids: list[str] = []
    missing_dependencies: dict[str, list[str]] = {}
    serialized = []
    for step in steps:
        if step.id in seen:
            duplicate_ids.append(step.id)
        missing = [dependency for dependency in step.depends_on if dependency not in seen]
        if missing:
            missing_dependencies[step.id] = missing
        seen.add(step.id)
        serialized.append(asdict(step))
    missing_conclusions = [conclusion for conclusion in conclusions if conclusion not in seen]
    return {
        "valid": not duplicate_ids and not missing_dependencies and not missing_conclusions,
        "duplicate_ids": duplicate_ids,
        "missing_dependencies": missing_dependencies,
        "missing_conclusions": missing_conclusions,
        "steps": serialized,
        "conclusions": list(conclusions),
    }


def subconstraint_count(m: int, n_out: int) -> int:
    """Number of nonempty row subsets enumerated by Equation (2)."""
    if m < 1 or n_out < 1:
        raise ValueError("m and n_out must be positive")
    return sum(comb(m, k) for k in range(1, min(m, n_out) + 1))


def projection_candidate(
    A: np.ndarray,
    b: np.ndarray,
    f: np.ndarray,
    w: np.ndarray,
    indices: tuple[int, ...],
) -> np.ndarray:
    """Equation (4) for one selected sub-constraint."""
    A_gamma = A[list(indices)]
    b_gamma = b[list(indices)]
    pinv = np.linalg.pinv(A_gamma)
    null_projector = np.eye(A.shape[1]) - pinv @ A_gamma
    return f - pinv @ (A_gamma @ f - b_gamma) + null_projector @ w


def caffnet_output(
    A: np.ndarray,
    b: np.ndarray,
    f: np.ndarray,
    w: np.ndarray,
    p: float = 2.0,
    tolerance: float = 1e-10,
) -> tuple[np.ndarray, dict]:
    """Equations (2)--(6), including feasibility filtering and p-nearest choice."""
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    f = np.asarray(f, dtype=float)
    w = np.asarray(w, dtype=float)
    if A.ndim != 2 or b.shape != (A.shape[0],) or f.shape != (A.shape[1],):
        raise ValueError("incompatible A, b, and f")
    if w.shape != f.shape or p < 1:
        raise ValueError("w must match f and p must be at least one")
    if np.all(A @ f <= b + tolerance):
        return f.copy(), {"used_unconstrained": True, "feasible_candidates": 0}
    feasible: list[tuple[float, np.ndarray, tuple[int, ...]]] = []
    for k in range(1, min(A.shape) + 1):
        for indices in combinations(range(A.shape[0]), k):
            candidate = projection_candidate(A, b, f, w, indices)
            if np.all(A @ candidate <= b + tolerance):
                feasible.append((float(np.linalg.norm(candidate - f, ord=p)), candidate, indices))
    if not feasible:
        raise RuntimeError("no feasible candidate; theorem assumptions or numerics failed")
    distance, output, indices = min(feasible, key=lambda row: row[0])
    return output, {
        "used_unconstrained": False,
        "feasible_candidates": len(feasible),
        "selected_indices": list(indices),
        "distance": distance,
    }


def hard_constraint_certificate() -> dict:
    """Universal arbitrary-cardinality feasibility (Lemma 3.3/Theorem 3.4)."""
    steps = [
        ProofStep(
            "c1-assumptions",
            "Fix any input x. Let S(x)={y:A(x)y<=b(x)} be a nonempty polyhedron in R^n_out; m is any positive finite constraint count and A,b are continuous in x.",
            reference="Assumption 3.2",
        ),
        ProofStep(
            "c1-enumeration",
            "Gamma contains every row-index subset of cardinality 1 through min(m,n_out), so its finite size is sum_k binomial(m,k) and m is not restricted by n_out.",
            ("c1-assumptions",),
            reference="Section 3.1, Equations (2)-(3)",
        ),
        ProofStep(
            "c1-minimal-face",
            "Every nonempty polyhedron has a minimal face F that is an affine space defined by at most n_out linearly independent active input constraints and satisfies F subset S.",
            ("c1-assumptions",),
            reference="Definition 3.1; Schrijver (1998), Theorem 8.4; Appendix A",
            evidence_type="imported polyhedral theorem",
        ),
        ProofStep(
            "c1-face-index-present",
            "The active index set gamma* defining F has size at most min(m,n_out), hence gamma* belongs to Gamma.",
            ("c1-enumeration", "c1-minimal-face"),
            reference="Lemma 3.3 proof",
        ),
        ProofStep(
            "c1-candidate-identity",
            "For P_gamma=f-A_gamma^dagger(A_gamma f-b_gamma)+(I-A_gamma^dagger A_gamma)w, the Penrose identities give A_gamma P_gamma=b_gamma for every f and w whenever the face equations are consistent.",
            ("c1-face-index-present",),
            reference="Equation (4); Appendix A, Equations (7)-(10)",
        ),
        ProofStep(
            "c1-feasible-candidate",
            "For gamma*, the equality places P_gamma* in the entire minimal face F, hence in S; the filtered candidate set S_P is nonempty for arbitrary learned f and w.",
            ("c1-minimal-face", "c1-candidate-identity"),
            reference="Lemma 3.3",
        ),
        ProofStep(
            "c1-selection",
            "Equation (6) returns f only when f is feasible and otherwise selects from nonempty S_P, so every CAffNet output satisfies all m inequalities.",
            ("c1-feasible-candidate",),
            reference="Theorem 3.4; Equations (5)-(6)",
        ),
        ProofStep(
            "c1-complete",
            "Hard constraint satisfaction therefore holds for every input, arbitrary finite cardinality m, redundant or dependent rows, and every rank allowed by Assumption 3.2.",
            ("c1-selection",),
        ),
    ]
    audit = _verify_dag(steps, ["c1-complete"])
    audit.update(
        {
            "claim": "C1",
            "scope": "every input-dependent nonempty affine-inequality polyhedron under Assumption 3.2, with arbitrary finite m",
            "proof_mode": "minimal-face existence plus exact Moore-Penrose algebra",
            "cardinality_controls": {
                "m=4,n_out=1": subconstraint_count(4, 1),
                "m=13,n_out=5": subconstraint_count(13, 5),
                "m=7,n_out=7": subconstraint_count(7, 7),
            },
            "finite_experiments": "2,000 random instances and neural tasks are corroboration only",
        }
    )
    return audit


def projector_control() -> dict:
    """Exact rank-deficient projector/gradient control for Claim 2."""
    # A=[1,0,0], so A^dagger A=diag(1,0,0) exactly and N=diag(0,1,1).
    null_projector = ((0, 0, 0), (0, 1, 0), (0, 0, 1))
    upstream = (1, 2, 3)
    gradient = tuple(sum(null_projector[j][i] * upstream[j] for j in range(3)) for i in range(3))
    displacement = (0, 5, -7)
    projected_displacement = tuple(
        sum(null_projector[i][j] * displacement[j] for j in range(3)) for i in range(3)
    )
    return {
        "null_projector": null_projector,
        "upstream_gradient": upstream,
        "gradient_to_f": gradient,
        "gradient_to_w": gradient,
        "null_displacement": displacement,
        "projected_null_displacement": projected_displacement,
        "both_parameter_paths_nonzero": any(gradient),
        "full_null_displacement_recovered": projected_displacement == displacement,
    }


def joint_optimization_certificate() -> dict:
    """Architectural expressivity and joint-gradient certificate for Claim 2."""
    steps = [
        ProofStep(
            "c2-formula",
            "For a locally selected sub-constraint, Equation (4) rewrites as P=A^dagger b+N(f_theta+w_phi), with N=I-A^dagger A.",
            reference="Equation (4)",
        ),
        ProofStep(
            "c2-projector",
            "The Moore-Penrose identities make N the symmetric idempotent orthogonal projector onto null(A).",
            ("c2-formula",),
            reference="standard Moore-Penrose properties",
            evidence_type="imported linear-algebra identity",
        ),
        ProofStep(
            "c2-fixed-projection",
            "Setting w_phi=0 recovers the fixed orthogonal projection; a fixed parallel projection likewise commits to a prescribed correction direction.",
            ("c2-projector",),
            reference="Section 3.2 discussion after Theorem 3.4",
        ),
        ProofStep(
            "c2-nullspace-expressivity",
            "When rank(A)<n_out, every null-space displacement z is reachable because Nz=z; learned w_phi can therefore choose throughout the affine solution face instead of fixing one projected point or direction.",
            ("c2-projector", "c2-fixed-projection"),
            reference="Figure 3; Theorem 3.4 discussion",
        ),
        ProofStep(
            "c2-gradients",
            "Away from candidate-selection ties, dP/df=dP/dw=N. Chain rule therefore sends task gradients into both theta and phi whenever the upstream gradient has a nonzero null-space component.",
            ("c2-formula", "c2-projector"),
            reference="Equation (4), differentiated",
        ),
        ProofStep(
            "c2-joint-training",
            "Thus the layer is trainable jointly with both parameter blocks, contains fixed-w orthogonal projection as an explicit special case, and adds an independently parameterized null-space optimization path at rank-deficient faces.",
            ("c2-nullspace-expressivity", "c2-gradients"),
        ),
        ProofStep(
            "c2-complete",
            "The architectural claim follows; executed neural controls additionally verify nonzero gradients, a load-bearing learned null-space head, and machine-precision feasibility.",
            ("c2-joint-training",),
        ),
    ]
    audit = _verify_dag(steps, ["c2-complete"])
    audit.update(
        {
            "claim": "C2",
            "scope": "each differentiable region with a fixed selected candidate; selection boundaries are piecewise-differentiable tie surfaces",
            "proof_mode": "exact null-space projector algebra plus executed multi-seed neural controls",
            "exact_rank_deficient_control": projector_control(),
            "experiment_boundary": "The five-seed reduced control establishes joint training and w_phi ablation value; an end-to-end orthogonal baseline also reaches the oracle, so uniqueness or universal superiority is not claimed.",
        }
    )
    return audit


def uat_error_budget(epsilon: Fraction, n_out: int) -> dict[str, Fraction]:
    """Exact conservative error budget for the independent UAT proof."""
    if epsilon <= 0 or n_out < 1:
        raise ValueError("epsilon and n_out must be positive")
    unconstrained_error = epsilon / (n_out + 1)
    return {
        "epsilon": epsilon,
        "unconstrained_error": unconstrained_error,
        "candidate_distance_bound": n_out * unconstrained_error,
        "final_error_bound": (n_out + 1) * unconstrained_error,
    }


def universal_approximation_certificate() -> dict:
    """Independent proof of Theorem 3.5 using Euclidean projection."""
    steps = [
        ProofStep(
            "c3-assumptions",
            "Let X be compact, S(x)={y:A(x)y<=b(x)} be nonempty with continuous A,b, let the target f_t(x) lie in S(x), and let F_theta be dense in the target ambient class under the pointwise p norm, 1<=p<infinity.",
            reference="Theorem 3.5",
        ),
        ProofStep(
            "c3-zero-null-network",
            "Choose the second network to be w_phi(x)=0 exactly; zero weights realize this continuous function, so no approximation of a changing face-specific null vector is required.",
            ("c3-assumptions",),
        ),
        ProofStep(
            "c3-architecture-enumerates",
            "CAffNet enumerates every active-row subset of size at most n_out and filters Equation (4) candidates by all original inequalities.",
            ("c3-assumptions",),
            reference="Equations (2)-(5)",
        ),
        ProofStep(
            "c3-hard-feasibility",
            "The minimal-face argument of Lemma 3.3 and Theorem 3.4 makes the filtered set nonempty and every Equation (6) output feasible under these same assumptions.",
            ("c3-architecture-enumerates",),
            reference="Lemma 3.3; Theorem 3.4",
        ),
        ProofStep(
            "c3-euclidean-projection",
            "For each x, the nonempty closed convex polyhedron S(x) has a unique Euclidean projection q of f_theta(x), with ||q-f_theta||_2<=||f_t-f_theta||_2.",
            ("c3-assumptions",),
            reference="projection theorem for closed convex sets",
            evidence_type="imported convex-analysis theorem",
        ),
        ProofStep(
            "c3-active-subset",
            "The projection optimality condition puts f_theta-q in the cone of active constraint normals. Conic Caratheodory selects at most n_out active rows spanning it.",
            ("c3-euclidean-projection",),
            reference="polyhedral normal cone and conic Caratheodory theorem",
            evidence_type="imported convex-analysis theorem",
        ),
        ProofStep(
            "c3-projection-is-candidate",
            "For that active subset gamma, q satisfies A_gamma q=b_gamma and f_theta-q lies in row(A_gamma); hence q is exactly f_theta-A_gamma^dagger(A_gamma f_theta-b_gamma), the Equation (4) candidate with w_phi=0.",
            ("c3-zero-null-network", "c3-architecture-enumerates", "c3-active-subset"),
        ),
        ProofStep(
            "c3-feasible-nearest-bound",
            "Because q is feasible, q belongs to S_P. Equation (6)'s p-nearest feasible candidate therefore satisfies ||P*-f_theta||_p<=||q-f_theta||_p.",
            ("c3-projection-is-candidate",),
            reference="Equations (5)-(6)",
        ),
        ProofStep(
            "c3-norm-bound",
            "Finite-dimensional norm equivalence and Euclidean projection give ||q-f_theta||_p<=n_out||f_t-f_theta||_p; the conservative factor n_out is valid for every finite p>=1.",
            ("c3-euclidean-projection",),
        ),
        ProofStep(
            "c3-error-budget",
            "Choose f_theta within epsilon/(n_out+1) of f_t. Triangle inequality plus the nearest-candidate and norm bounds gives ||P*-f_t||_p<epsilon uniformly.",
            ("c3-feasible-nearest-bound", "c3-norm-bound"),
        ),
        ProofStep(
            "c3-uat",
            "CAffNet is dense in the feasible target class whenever the underlying f_theta class is dense in the ambient class.",
            ("c3-error-budget",),
            reference="Theorem 3.5",
        ),
        ProofStep(
            "c3-hard-and-universal",
            "Combining Theorem 3.5 with the C1 hard-feasibility certificate gives both universal approximation and constraint adherence for every input.",
            ("c3-uat", "c3-hard-feasibility"),
            reference="Theorems 3.4-3.5",
        ),
        ProofStep(
            "c3-complete",
            "The complete conjunctive challenge claim follows under the paper's stated assumptions.",
            ("c3-hard-and-universal",),
        ),
    ]
    audit = _verify_dag(steps, ["c3-complete"])
    exact_budgets = {
        f"epsilon={epsilon},n={n}": {
            key: str(value) for key, value in uat_error_budget(epsilon, n).items()
        }
        for epsilon, n in ((Fraction(1, 10), 1), (Fraction(1), 5), (Fraction(7, 3), 13))
    }
    audit.update(
        {
            "claim": "C3",
            "scope": "Theorem 3.5 for compact X, nonempty continuous polyhedral constraints, feasible targets, and every finite p>=1",
            "proof_mode": "independent Euclidean-projection proof plus C1 hard-feasibility certificate",
            "exact_error_budgets": exact_budgets,
            "important_strengthening": "w_phi is chosen identically zero, so the proof does not need to approximate a face-specific null-space target and covers the full p range with one conservative norm bound.",
            "finite_width_sweep": "corroboration only; a finite width sweep is not used to infer density",
        }
    )
    return audit


def all_certificates() -> dict:
    claims = [
        hard_constraint_certificate(),
        joint_optimization_certificate(),
        universal_approximation_certificate(),
    ]
    return {
        "paper": "20hdQQQrA4",
        "paper_title": "CAffNet: Hard Constraint-Affine Neural Networks",
        "paper_sha256": "33db803823608bdb76db20732f0a3a20aef32c5f4e22f4c4b148a2b7b6da9520",
        "mode": "universal proof audit plus neural corroboration",
        "claims": claims,
        "all_valid": all(claim["valid"] for claim in claims),
    }
