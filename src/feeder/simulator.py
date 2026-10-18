"""Solve one named feeder through an isolated DSS-Extensions context.

This module owns physical load and regulator writes, snapshot power flows, and
feeder-specific voltage measurement selection. Observation encoding, rewards,
and operational metrics remain in ``common``.
"""

import dataclasses

import numpy as np

from feeder._dss import DssBackend
from feeder.bands import LIVE_NODE_MIN_V
from feeder.config import read_config
from feeder.taps import (
    N_TAPS,
    TAP_MAX,
    TAP_MIN,
    index_of_ratio,
    neutral_positions,
    ratio_of_index,
    validate_positions,
)

TAP_WINDING = 2


class SimulationError(RuntimeError):
    """Report an invalid feeder case or failed simulation operation."""


@dataclasses.dataclass(frozen=True)
class SolveResult:
    """Hold one power-flow result as independent, read-only physical arrays.

    Voltages follow ``node_names`` in the feeder's fixed measurement order and
    use p.u. Tap positions and ratios follow controlled-regulator order.
    ``vmin`` and ``vmax`` summarize the selected voltage measurements.
    """

    converged: bool
    node_names: tuple
    voltages: np.ndarray
    tap_positions: np.ndarray
    tap_ratios: np.ndarray
    vmin: float
    vmax: float


def _readonly(values, dtype):
    result = np.array(values, dtype=dtype, copy=True)
    result.flags.writeable = False
    return result


def _loads(values, name, count):
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain finite floats") from error
    if array.ndim != 1 or array.shape != (count,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be one-dimensional with {count} finite floats")
    return np.array(array, dtype=np.float64, copy=True)


class FeederSimulator:
    """Own one named feeder context and its fixed experiment interfaces.

    Construction compiles the case, freezes load and measurement order, checks
    the supported regulator grid, records nominal kW and kvar, and completes a
    neutral fixed-tap initialization solve. Every solve writes all loads before
    snapshot execution; fixed-action and persistent native solves also write
    all taps, while the from-case native solve lets the compiled case set them.
    Returned arrays do not alias engine state.
    """

    def __init__(self, feeder="ieee13", backend=None):
        """Initialize and validate one named feeder in an isolated engine context."""
        self._config = read_config(feeder)
        self.feeder = self._config.name
        self._case_path = self._config.case_path
        self.regulator_names = self._config.regulator_names
        self._windings = self._config.windings
        self.action_count = len(self.regulator_names)
        self.neutral_positions = neutral_positions(self.action_count)
        self._backend = DssBackend() if backend is None else backend
        self._backend.compile_case(self._case_path)
        names = tuple(self._backend.load_names())
        if self._config.loads is not None and names != self._config.loads:
            raise SimulationError(f"compiled load names {names} do not match {self._config.loads}")
        if not names or len(set(names)) != len(names):
            raise SimulationError(f"compiled load names are invalid: {names}")
        self.load_names = names
        self._check_equipment()
        nominal = [self._backend.read_load(name) for name in names]
        if not np.isfinite(nominal).all():
            raise SimulationError("nominal loads must be finite")
        self._nominal_kw = _readonly([item[0] for item in nominal], np.float64)
        self._nominal_kvar = _readonly([item[1] for item in nominal], np.float64)
        self._engine_version = self._backend.version()
        self._backend.set_tap_ratios(self.regulator_names, ratio_of_index(self.neutral_positions))
        self._backend.set_control_mode("OFF")
        if not self._backend.solve_snapshot():
            raise SimulationError("initialization solve did not converge")
        self._freeze_measurements()

    @property
    def case_path(self):
        """Return the configured feeder case path."""
        return self._case_path

    @property
    def engine_version(self):
        """Return the DSS-Extensions version captured at construction."""
        return self._engine_version

    @property
    def nominal_kw(self):
        """Return a copy of nominal active loads in compiled load order."""
        return self._nominal_kw.copy()

    @property
    def nominal_kvar(self):
        """Return a copy of nominal reactive loads in compiled load order."""
        return self._nominal_kvar.copy()

    @property
    def measurement_names(self):
        """Return the fixed ordered identifiers for reported voltage measurements."""
        return self._measurement_names

    @property
    def node_names(self):
        """Return the measurement identifiers under the historical property name."""
        return self._measurement_names

    def _check_equipment(self):
        """Require configured regulators on the supported 33-position grid."""
        installed = {name.lower() for name in self._backend.transformer_names()}
        for name, winding in zip(self.regulator_names, self._windings, strict=True):
            if name.lower() not in installed:
                raise SimulationError(f"controlled transformer {name!r} is missing")
            minimum, maximum, intervals = self._backend.transformer_grid(name, winding)
            if not (
                np.isclose(minimum, TAP_MIN, rtol=0.0, atol=1e-9)
                and np.isclose(maximum, TAP_MAX, rtol=0.0, atol=1e-9)
                and intervals == N_TAPS - 1
            ):
                raise SimulationError(
                    f"{name} tap grid {minimum:g}..{maximum:g} with {intervals} "
                    "intervals differs from the supported grid"
                )

    def _check_recompile(self):
        """Restore post-compile invariants needed by a native-control solve."""
        names = tuple(self._backend.load_names())
        if names != self.load_names:
            raise SimulationError(f"compiled load names changed from {self.load_names} to {names}")
        self._check_equipment()
        if self._config.native_control_iterations is not None:
            self._backend.set_native_control_iterations(self._config.native_control_iterations)

    def _freeze_measurements(self):
        """Select and freeze the feeder's ordered voltage measurement set.

        IEEE13 retains initially live line-ground nodes. The IEEE37 variants
        retain three line-line phase pairs for each three-phase non-source bus.
        """
        if self._config.voltage_convention == "line-ground":
            names = tuple(self._backend.node_names())
            voltages = np.asarray(self._backend.node_voltages_pu())
            if len(names) != len(voltages):
                raise SimulationError("node names and voltages have different lengths")
            if not np.isfinite(voltages).all():
                raise SimulationError("initialization voltages must be finite")
            self._all_node_names = names
            self._node_count = len(names)
            self._mask = np.flatnonzero(voltages > LIVE_NODE_MIN_V)
            if not len(self._mask):
                raise SimulationError("initialization solve has no live node")
            self._measurement_names = tuple(names[index] for index in self._mask)
            return
        names, voltages, bases = self._backend.line_line_measurements()
        if not names or len(names) != len(voltages) or len(names) != len(bases):
            raise SimulationError("line-line measurement names and values have different lengths")
        if not np.isfinite(voltages).all() or not np.isfinite(bases).all() or (bases <= 0).any():
            raise SimulationError("line-line voltages and bases must be finite and positive")
        if len(set(names)) != len(names):
            raise SimulationError("line-line measurement names must be unique")
        self._measurement_names = names
        self._line_bases = _readonly(bases, np.float64)

    def _measure(self, converged):
        """Read selected p.u. voltages and verify their frozen identity and bases."""
        if self._config.voltage_convention == "line-ground":
            names = tuple(self._backend.node_names())
            voltages = np.asarray(self._backend.node_voltages_pu())
            if names != self._all_node_names:
                raise SimulationError("node names changed during the solve")
            if len(voltages) != self._node_count:
                raise SimulationError("node count changed during the solve")
            selected = voltages[self._mask]
        else:
            names, selected, bases = self._backend.line_line_measurements()
            if names != self._measurement_names:
                raise SimulationError("line-line measurement names changed during the solve")
            if len(selected) != len(names) or len(bases) != len(names):
                raise SimulationError("line-line measurement count changed during the solve")
            if not np.array_equal(bases, self._line_bases):
                raise SimulationError("line-line voltage bases changed during the solve")
        if converged and not np.isfinite(selected).all():
            raise SimulationError("converged solve voltages must be finite")
        return _readonly(selected, np.float64)

    def _write_loads(self, kw, kvar):
        """Write absolute kW and kvar arrays in the frozen compiled load order."""
        for name, active, reactive in zip(self.load_names, kw, kvar, strict=True):
            self._backend.write_load(name, active, reactive)

    def _read_taps(self):
        """Read winding-two ratios in controlled-regulator order."""
        return np.array(
            [
                self._backend.read_tap_ratio(name, winding)
                for name, winding in zip(self.regulator_names, self._windings, strict=True)
            ],
            dtype=np.float64,
        )

    def _capture(self, converged, expected_positions=None):
        """Validate engine readback and copy one solve into a physical result."""
        tap_ratios = self._read_taps()
        tap_positions = index_of_ratio(tap_ratios)
        expected_ratios = ratio_of_index(tap_positions)
        if not np.allclose(tap_ratios, expected_ratios, rtol=0.0, atol=1e-9):
            raise SimulationError(f"tap readback {tap_ratios} is not on the physical grid")
        if expected_positions is not None and not np.array_equal(tap_positions, expected_positions):
            raise SimulationError(f"tap readback {tap_positions} differs from {expected_positions}")
        voltages = self._measure(converged)
        return SolveResult(
            bool(converged),
            self._measurement_names,
            voltages,
            _readonly(tap_positions, int),
            _readonly(tap_ratios, np.float64),
            float(voltages.min()),
            float(voltages.max()),
        )

    def _checked_inputs(self, kw, kvar):
        count = len(self.load_names)
        return _loads(kw, "kw", count), _loads(kvar, "kvar", count)

    def solve(self, kw, kvar, tap_positions):
        """Run a fixed-action snapshot power flow.

        ``kw`` and ``kvar`` are absolute load arrays in ``load_names`` order.
        ``tap_positions`` supplies one legal integer position per controlled
        regulator. The call disables native controls and returns the requested
        grid positions, physical ratios, selected p.u. voltages, and convergence.
        A nonconverged engine result is returned when its measurements remain
        structurally valid; callers decide whether to reject it.
        """
        active, reactive = self._checked_inputs(kw, kvar)
        positions = validate_positions(tap_positions, self.action_count)
        self._write_loads(active, reactive)
        self._backend.set_tap_ratios(self.regulator_names, ratio_of_index(positions))
        self._backend.set_control_mode("OFF")
        return self._capture(self._backend.solve_snapshot(), positions)

    def _native_solve(self):
        """Run native controls and capture their settled grid positions."""
        try:
            converged = self._backend.solve_snapshot()
        except Exception as error:
            raise SimulationError(f"native control solve failed: {error}") from error
        if converged and not self._backend.control_actions_done():
            raise SimulationError("native control actions did not complete")
        return self._capture(converged)

    def solve_native_from_case(self, kw, kvar):
        """Let the compiled case initialize and select native RegControl taps.

        The call recompiles the case, validates its load and regulator identity,
        writes absolute loads, and preserves the case's own control setup. It is
        the native-selection route used by static reference evaluation.
        """
        active, reactive = self._checked_inputs(kw, kvar)
        self._backend.compile_case(self._case_path)
        self._check_recompile()
        self._write_loads(active, reactive)
        return self._native_solve()

    def solve_native(self, kw, kvar, start_tap_positions):
        """Run native RegControl from supplied persistent tap positions.

        The call recompiles the case, writes absolute loads and starting taps,
        enables static native control, and returns the settled physical result.
        Sequential reference evaluation uses this route from the previous
        hour's executed positions.
        """
        active, reactive = self._checked_inputs(kw, kvar)
        positions = validate_positions(start_tap_positions, self.action_count)
        self._backend.compile_case(self._case_path)
        self._check_recompile()
        self._write_loads(active, reactive)
        self._backend.set_tap_ratios(self.regulator_names, ratio_of_index(positions))
        self._backend.set_control_mode("STATIC")
        return self._native_solve()
