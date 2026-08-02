
from itertools import combinations

import numpy as np
import colour

import fitness as _fitness

_ORIGINALS = {}


def _pairwise_delta_e(palette_lab):
    """Vectorised twin of fitness.pairwise_delta_e."""
    palette_lab = np.asarray(palette_lab, dtype=np.float64)
    k = len(palette_lab)
    pairs = list(combinations(range(k), 2))
    if not pairs:
        return np.array([]), pairs

    i_idx = [i for i, _ in pairs]
    j_idx = [j for _, j in pairs]
    distances = np.atleast_1d(
        colour.delta_E(palette_lab[i_idx], palette_lab[j_idx], method="CIE 2000")
    )
    return distances, pairs


def _fidelity_violations(original_lab, candidate_lab, tau=_fitness.DEFAULT_TAU):
    """Vectorised twin of fitness.fidelity_violations."""
    original_lab = np.asarray(original_lab, dtype=np.float64)
    candidate_lab = np.asarray(candidate_lab, dtype=np.float64)
    drifts = np.atleast_1d(
        colour.delta_E(original_lab, candidate_lab, method="CIE 2000")
    )
    violations = np.maximum(0.0, drifts - tau)
    return drifts, float(np.sum(violations))


def enable():
    """Swap in the vectorised versions. Safe to call more than once."""
    if _ORIGINALS:
        return
    _ORIGINALS["pairwise_delta_e"] = _fitness.pairwise_delta_e
    _ORIGINALS["fidelity_violations"] = _fitness.fidelity_violations
    _fitness.pairwise_delta_e = _pairwise_delta_e
    _fitness.fidelity_violations = _fidelity_violations


def disable():
    """Restore the original loop-based versions."""
    for attr, func in _ORIGINALS.items():
        setattr(_fitness, attr, func)
    _ORIGINALS.clear()