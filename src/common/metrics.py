"""Assess fixed operational voltage metrics independently of training rewards."""

import numpy as np

from feeder.bands import PRIMARY_BAND


def voltage_metrics(voltages, *, converged=True):
    """Assess physical p.u. measurements against the inclusive 0.95--1.05 band.

    Cost is the mean squared excursion, with zero contribution in band.
    ``violation_count`` counts measurements outside the band. Feasibility means
    every measurement is in band and the power flow converged.
    """
    values = np.asarray(voltages, dtype=float)
    lo, hi = PRIMARY_BAND
    excursion = np.maximum(lo - values, 0.0) + np.maximum(values - hi, 0.0)
    return {
        "vmin": float(values.min()),
        "vmax": float(values.max()),
        "cost": float((excursion**2).mean()),
        "violation_count": int((excursion > 0).sum()),
        "feasible": bool((excursion == 0).all()) if converged else False,
    }
