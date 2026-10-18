"""Define learning rewards from executed physical outcomes.

These formulas affect replay and training only. Evaluation uses fixed operational
metrics from ``common.metrics`` even when a training reward changes.
"""

import numpy as np


def voltage_band_reward(
    voltages,
    *,
    mad_weight=0.0,
    voltage_min=0.95,
    voltage_max=1.05,
    scale=1000.0,
):
    """Return the negative scaled one-shot learning cost.

    For each monitored physical p.u. voltage, the band term is its distance below
    ``voltage_min`` or above ``voltage_max``, and zero in band. The function
    averages squared band terms across measurements. It optionally adds
    ``mad_weight`` times the mean absolute deviation from 1.0 p.u., including
    in-band measurements, then multiplies the total by ``-scale``.
    """
    values = np.asarray(voltages, dtype=float)
    excursion = np.maximum(voltage_min - values, 0.0) + np.maximum(values - voltage_max, 0.0)
    cost = float((excursion**2).mean())
    mean_abs_deviation = float(np.abs(values - 1.0).mean())
    return -scale * (cost + mad_weight * mean_abs_deviation)


def voltage_band_switching_reward(
    voltages,
    tap_steps,
    *,
    beta=1e-5,
    voltage_min=0.95,
    voltage_max=1.05,
    scale=1000.0,
):
    """Return the negative scaled voltage-plus-switching cost for one hour.

    The voltage term averages squared band excursions across monitored physical
    p.u. measurements; in-band values contribute zero. ``tap_steps`` sums
    absolute executed position changes across regulators. It includes the first
    movement from neutral and measures distance, not switching-event count. The
    reward is ``-scale * (voltage_cost + beta * tap_steps)``.
    """
    values = np.asarray(voltages, dtype=float)
    excursion = np.maximum(voltage_min - values, 0.0) + np.maximum(values - voltage_max, 0.0)
    cost = float((excursion**2).mean())
    total = float(cost + beta * tap_steps)
    return -scale * total
