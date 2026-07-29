"""
Fitness Function Module
=======================
Implements the constrained optimization formulation:

    maximize   min{ ΔE_CVD(c'_i, c'_j) }   for all i < j
    subject to ΔE(c_i, c'_i) ≤ τ           for all i

Where:
    - P  = {c₁, ..., cₖ}  is the original palette in CIELAB
    - P' = {c'₁, ..., c'ₖ} is the candidate (adjusted) palette in CIELAB
    - ΔE_CVD(c'_i, c'_j) is the CIEDE2000 distance between colors i and j
      as perceived under a CVD simulation
    - τ is the maximum allowed perceptual drift from the original color
      (design fidelity constraint)

The fitness value is the minimum pairwise ΔE across all CVD types being
considered. Higher is better — the optimizer is maximizing.

Constraint violations are handled via a penalty term so the function can
be used with both constrained and unconstrained optimizers.
"""

import numpy as np
from itertools import combinations
from cvd_module import simulate_palette_lab, delta_e


# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------
DEFAULT_TAU = 12.0        # max ΔE drift allowed per color (fidelity threshold)
PENALTY_WEIGHT = 50.0     # penalty per unit of constraint violation
DEFAULT_CVD_TYPES = ["protan", "deutan", "tritan"]


# ---------------------------------------------------------------------------
# Core pairwise ΔE computation
# ---------------------------------------------------------------------------
def pairwise_delta_e(palette_lab):
    """
    Compute all pairwise CIEDE2000 distances for a palette.

    Parameters
    ----------
    palette_lab : ndarray, shape (k, 3)
        Palette in CIELAB.

    Returns
    -------
    distances : ndarray, shape (n_pairs,)
        ΔE for each unique pair.
    pairs : list of (int, int)
        The corresponding index pairs.
    """
    k = len(palette_lab)
    pairs = list(combinations(range(k), 2))
    distances = np.array([
        delta_e(palette_lab[i], palette_lab[j])
        for i, j in pairs
    ])
    return distances, pairs


# ---------------------------------------------------------------------------
# Fidelity constraint check
# ---------------------------------------------------------------------------
def fidelity_violations(original_lab, candidate_lab, tau=DEFAULT_TAU):
    """
    Check how much each color has drifted from its original.

    Returns
    -------
    drifts : ndarray, shape (k,)
        ΔE between each original and candidate color.
    total_violation : float
        Sum of max(0, drift - τ) across all colors. Zero means feasible.
    """
    drifts = np.array([
        delta_e(original_lab[i], candidate_lab[i])
        for i in range(len(original_lab))
    ])
    violations = np.maximum(0, drifts - tau)
    return drifts, float(np.sum(violations))

# ---------------------------------------------------------------------------
# Detect problematic colour pairs
# ---------------------------------------------------------------------------
def find_problematic_pairs(original_lab,
                           threshold=10,
                           cvd_types=None,
                           return_details=False):
    """
    Find all colour pairs whose simulated ΔE falls below the threshold.

    If return_details=True, also return detailed information for printing.
    """

    if cvd_types is None:
        cvd_types = DEFAULT_CVD_TYPES

    problematic_pairs = set()
    details = []

    for cvd in cvd_types:

        simulated = simulate_palette_lab(
            original_lab,
            cvd_type=cvd,
            severity=1.0
        )

        distances, pairs = pairwise_delta_e(simulated)

        for dE, pair in zip(distances, pairs):

            if dE < threshold:

                problematic_pairs.add(pair)

                details.append({
                    "pair": pair,
                    "cvd": cvd,
                    "deltaE": float(dE)
                })

    if return_details:
        return problematic_pairs, details

    return problematic_pairs

def print_conflict_report(original_lab,
                          optimized_lab=None,
                          threshold=10,
                          cvd_types=None):
    """
    Print every conflicting colour pair.

    If optimized_lab is provided, also show the ΔE after optimisation
    for the same pair and whether it was fixed.
    """

    if cvd_types is None:
        cvd_types = ["protan", "deutan", "tritan"]

    _, details = find_problematic_pairs(
        original_lab,
        threshold=threshold,
        cvd_types=cvd_types,
        return_details=True
    )

    print("\n" + "=" * 70)
    print("CONFLICT REPORT")
    print("=" * 70)

    if not details:
        print("No conflicting colour pairs found.")
        print("=" * 70)
        return

    # Sort from worst conflict to least
    details = sorted(details, key=lambda x: x["deltaE"])

    for item in details:

        i, j = item["pair"]
        cvd = item["cvd"]

        before = item["deltaE"]

        line = (
            f"Colours ({i}, {j})"
            f"  |  {cvd.upper():7s}"
            f"  |  Before = {before:6.2f}"
        )

        if optimized_lab is not None:

            simulated = simulate_palette_lab(
                optimized_lab,
                cvd_type=cvd,
                severity=1.0
            )

            after = delta_e(
                simulated[i],
                simulated[j]
            )

            if after >= threshold:
                status = "✓ FIXED"
            else:
                status = "✗ STILL CONFLICT"

            line += (
                f"  |  After = {after:6.2f}"
                f"  |  Δ = {after-before:+5.2f}"
                f"  |  {status}"
            )

        print(line)

    print("=" * 70)


def print_colour_summary(original_lab,
                         threshold=10,
                         cvd_types=None,
                         mutable_indices=None):
    """
    Print a summary showing how many conflicts each colour participates in.
    """

    _, details = find_problematic_pairs(
        original_lab,
        threshold=threshold,
        cvd_types=cvd_types,
        return_details=True
    )

    n_colours = len(original_lab)

    summary = {
        i: {
            "count": 0,
            "partners": set(),
            "cvds": set()
        }
        for i in range(n_colours)
    }

    for item in details:

        i, j = item["pair"]

        summary[i]["count"] += 1
        summary[j]["count"] += 1

        summary[i]["partners"].add(j)
        summary[j]["partners"].add(i)

        summary[i]["cvds"].add(item["cvd"])
        summary[j]["cvds"].add(item["cvd"])

    print("\n" + "=" * 70)
    print("COLOUR SUMMARY")
    print("=" * 70)

    print(f"{'Colour':<8}"
          f"{'Conflicts':<12}"
          f"{'With':<18}"
          f"{'CVD':<22}"
          f"{'Optimised'}")

    print("-" * 70)

    for i in range(n_colours):

        partners = ", ".join(map(str, sorted(summary[i]["partners"])))
        cvds = ", ".join(sorted(summary[i]["cvds"]))

        if partners == "":
            partners = "-"

        if cvds == "":
            cvds = "-"

        selected = (
            "YES"
            if mutable_indices is not None and i in mutable_indices
            else "NO"
        )

        print(f"{i:<8}"
              f"{summary[i]['count']:<12}"
              f"{partners:<18}"
              f"{cvds:<22}"
              f"{selected}")

    print("=" * 70)

# ---------------------------------------------------------------------------
# Main fitness function
# ---------------------------------------------------------------------------
def fitness(candidate_lab, original_lab, cvd_types=None, tau=DEFAULT_TAU,
            penalty_weight=PENALTY_WEIGHT, return_details=False):
    """
    Compute fitness of a candidate palette.

    Fitness = min-pairwise-ΔE across all CVD simulations − penalty for
    fidelity violations.

    Parameters
    ----------
    candidate_lab : ndarray, shape (k, 3)
        Candidate palette in CIELAB.
    original_lab : ndarray, shape (k, 3)
        Original palette in CIELAB (for fidelity constraint).
    cvd_types : list of str, optional
        CVD types to evaluate. Default: all three.
    tau : float
        Fidelity threshold (max allowed ΔE drift per color).
    penalty_weight : float
        Multiplier for constraint violation penalty.
    return_details : bool
        If True, return a dict with full diagnostics.

    Returns
    -------
    float
        Fitness value (higher is better). Can be negative if heavily
        penalized.
    dict (only if return_details=True)
        Detailed breakdown of the evaluation.
    """
    if cvd_types is None:
        cvd_types = DEFAULT_CVD_TYPES

    # --- Fidelity constraint ---
    drifts, total_violation = fidelity_violations(original_lab, candidate_lab, tau)
    penalty = penalty_weight * total_violation

    # --- CVD distinguishability (the objective) ---
    min_de_per_cvd = {}
    worst_pair_per_cvd = {}

    for cvd in cvd_types:
        simulated = simulate_palette_lab(candidate_lab, cvd_type=cvd, severity=1.0)
        distances, pairs = pairwise_delta_e(simulated)
        min_idx = np.argmin(distances)
        min_de_per_cvd[cvd] = distances[min_idx]
        worst_pair_per_cvd[cvd] = pairs[min_idx]

    # The objective: worst-case min-ΔE across all CVD types
    objective = min(min_de_per_cvd.values())

    # Final fitness = objective − penalty
    score = objective - penalty

    if return_details:
        details = {
            "score": score,
            "objective": objective,           # min-ΔE across all CVDs
            "penalty": penalty,
            "total_violation": total_violation,
            "drifts": drifts,                 # per-color drift from original
            "min_de_per_cvd": min_de_per_cvd, # min-ΔE per CVD type
            "worst_pair_per_cvd": worst_pair_per_cvd,
            "feasible": total_violation == 0,
        }
        return score, details

    return score


# ---------------------------------------------------------------------------
# Utility: evaluate original (un-optimized) palette as baseline
# ---------------------------------------------------------------------------
def evaluate_baseline(original_lab, cvd_types=None):
    """
    Show how bad the original palette is under each CVD.
    Useful for before/after comparison.
    """
    if cvd_types is None:
        cvd_types = DEFAULT_CVD_TYPES

    print("Baseline evaluation (original palette, no optimization)")
    print("=" * 55)

    for cvd in cvd_types:
        simulated = simulate_palette_lab(original_lab, cvd_type=cvd, severity=1.0)
        distances, pairs = pairwise_delta_e(simulated)
        min_idx = np.argmin(distances)
        print(f"  {cvd.upper():6s}  min ΔE = {distances[min_idx]:6.2f}  "
              f"(worst pair: colors {pairs[min_idx]})")

    print("=" * 55)