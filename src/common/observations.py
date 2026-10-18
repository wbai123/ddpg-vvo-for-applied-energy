"""Encode feeder physical states as ordered float32 network observations.

These transformations do not alter power-flow state. Their size helpers derive
feeder-dependent network widths from the simulator's frozen interfaces.
"""

import numpy as np

from feeder.loads import HOURS
from feeder.taps import NEUTRAL_K


def voltage_observation(state):
    """Return ``(V - 1) / 0.1`` in the simulator's measurement order."""
    return ((np.asarray(state.voltages) - 1.0) / 0.1).astype(np.float32)


def voltage_observation_size(simulator):
    """Return the feeder-dependent width of the voltage observation."""
    return len(simulator.measurement_names)


def day_observation(state):
    """Encode voltage, held taps, and hour for a sequential decision.

    Features concatenate normalized voltages, one tap feature per controlled
    regulator, then sine and cosine of the current hour. Tap position ``k`` maps
    to ``(k - 16) / 16``, so legal positions 0, 16, and 32 map to -1, 0, and 1.
    The clock angle is ``2*pi*hour/24``. The result uses float32 throughout.
    """
    angle = 2.0 * np.pi * state.hour / HOURS
    taps = (state.tap_positions.astype(np.float32) - float(NEUTRAL_K)) / float(NEUTRAL_K)
    clock = np.array([np.sin(angle), np.cos(angle)], dtype=np.float32)
    return np.concatenate((voltage_observation(state), taps, clock)).astype(np.float32)


def day_observation_size(simulator):
    """Return measurement count plus regulator count plus two clock features."""
    return len(simulator.measurement_names) + simulator.action_count + 2
