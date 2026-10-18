"""Convert between discrete regulator positions and physical tap ratios.

All supported regulators use 33 positions on the 0.90--1.10 p.u. grid.
Movement is measured as distance in tap steps, not as a switching-event count.
"""

import numpy as np

TAP_MIN, TAP_MAX, TAP_STEP, N_TAPS = 0.90, 1.10, 0.00625, 33
NEUTRAL_K = 16
PHYSICAL_RANGE = (TAP_MIN, TAP_MAX)


def neutral_positions(n_controls):
    """Return an owned integer vector at the neutral position for each regulator."""
    return np.full(n_controls, NEUTRAL_K, dtype=int)


def validate_positions(k, n_controls):
    """Return an owned vector of legal integer positions in regulator order."""
    values = np.asarray(k)
    if (
        values.shape != (n_controls,)
        or values.dtype.kind not in "iu"
        or (values < 0).any()
        or (values >= N_TAPS).any()
    ):
        raise ValueError(f"tap positions {values} must be {n_controls} integers in 0..{N_TAPS - 1}")
    return np.array(values, dtype=int, copy=True)


def ratio_of_index(k):
    """Map tap positions to physical p.u. ratios, exact to five decimals."""
    return np.round(TAP_MIN + TAP_STEP * np.asarray(k, dtype=float), 5)


def index_of_ratio(ratio):
    """Project proposed p.u. ratios onto the legal discrete tap grid.

    Values outside 0.90--1.10 are clipped. An exact midpoint rounds to the
    higher position.
    """
    r = np.clip(np.asarray(ratio, dtype=float), TAP_MIN, TAP_MAX)
    # The offset corrects an exact midpoint that evaluates below 0.5.
    k = np.floor((r - TAP_MIN) / TAP_STEP + 0.5 + 1e-9).astype(int)
    return np.clip(k, 0, N_TAPS - 1)


def project(proposed):
    """Return executed tap positions and their p.u. ratios for a proposal."""
    k = index_of_ratio(proposed)
    return k, ratio_of_index(k)


def tap_steps(k_after, k_before):
    """Sum absolute position changes across regulators for one decision."""
    return int(np.abs(np.asarray(k_after, dtype=int) - np.asarray(k_before, dtype=int)).sum())
