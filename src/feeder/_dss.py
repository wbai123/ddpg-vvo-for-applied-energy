"""Adapt DSS-Extensions operations for the physical feeder simulator.

DSS-Extensions contexts are independent within one process. This module
owns engine-specific reads and writes but does not define experiment rewards,
observations, or metrics. It makes no thread-safety claim.
"""

import math

import numpy as np
import opendssdirect


class DssBackend:
    """Own one isolated DSS context and expose the simulator's engine operations."""

    def __init__(self):
        """Create a DSS context that cannot change the process working directory."""
        self._context = opendssdirect.NewContext()
        self._context.Basic.AllowChangeDir(False)

    def compile_case(self, path):
        """Compile a feeder case into this context."""
        self._context.Text.Command(f"compile '{path}'")

    def load_names(self):
        return tuple(self._context.Loads.AllNames())

    def transformer_names(self):
        return tuple(self._context.Transformers.AllNames())

    def transformer_grid(self, name, winding):
        """Return a transformer's minimum ratio, maximum ratio, and interval count."""
        self._context.Transformers.Name(name)
        self._context.Transformers.Wdg(winding)
        return (
            float(self._context.Transformers.MinTap()),
            float(self._context.Transformers.MaxTap()),
            int(self._context.Transformers.NumTaps()),
        )

    def read_load(self, name):
        """Return one compiled load's active kW and reactive kvar."""
        self._context.Loads.Name(name)
        return float(self._context.Loads.kW()), float(self._context.Loads.kvar())

    def write_load(self, name, kw, kvar):
        """Set one compiled load's absolute active kW and reactive kvar."""
        self._context.Loads.Name(name)
        self._context.Loads.kW(kw)
        self._context.Loads.kvar(kvar)

    def set_tap_ratios(self, regulators, ratios):
        """Write winding-two p.u. tap ratios in controlled-regulator order."""
        for regulator, ratio in zip(regulators, ratios):
            self._context.Text.Command(f"Transformer.{regulator}.Taps=[1.0 {ratio}]")

    def read_tap_ratio(self, name, winding):
        """Return one transformer's physical p.u. ratio on the selected winding."""
        self._context.Transformers.Name(name)
        self._context.Transformers.Wdg(winding)
        return float(self._context.Transformers.Tap())

    def set_control_mode(self, mode):
        """Select fixed taps (OFF) or native RegControl actions (STATIC)."""
        if mode not in {"OFF", "STATIC"}:
            raise ValueError(f"unsupported control mode {mode!r}")
        self._context.Text.Command(f"Set Controlmode={mode}")

    def set_native_control_iterations(self, limit):
        """Set the maximum iterations available to native controls."""
        self._context.Solution.MaxControlIterations(limit)

    def control_actions_done(self):
        """Report whether the native control queue settled during the solve."""
        return bool(self._context.Solution.ControlActionsDone())

    def solve_snapshot(self):
        """Run a snapshot power flow and report engine convergence."""
        self._context.Text.Command("Set mode=snapshot")
        self._context.Text.Command("Solve")
        return bool(self._context.Solution.Converged())

    def node_names(self):
        return tuple(self._context.Circuit.AllNodeNames())

    def node_voltages_pu(self):
        """Return all bus voltage magnitudes in engine node order and p.u."""
        return np.array(self._context.Circuit.AllBusMagPu(), dtype=np.float64, copy=True)

    def line_line_measurements(self):
        """Measure three-phase line pairs outside the source bus.

        Returns ordered pair identifiers, complex-difference magnitudes in p.u.,
        and their line-to-line voltage bases in volts.
        """
        names, values, bases = [], [], []
        for bus in self._context.Circuit.AllBusNames():
            if bus.lower() == "sourcebus":
                continue
            self._context.Circuit.SetActiveBus(bus)
            nodes = self._context.Bus.Nodes()
            if not all(phase in nodes for phase in (1, 2, 3)):
                continue
            raw = np.asarray(self._context.Bus.Voltages(), dtype=np.float64)
            complex_voltages = raw[::2] + 1j * raw[1::2]
            phases = {phase: complex_voltages[index] for index, phase in enumerate(nodes)}
            base = math.sqrt(3.0) * float(self._context.Bus.kVBase()) * 1000.0
            for label, first, second in (("AB", 1, 2), ("BC", 2, 3), ("CA", 3, 1)):
                names.append(f"{bus}.{label}")
                values.append(abs(phases[first] - phases[second]) / base)
                bases.append(base)
        return (
            tuple(names),
            np.asarray(values, dtype=np.float64),
            np.asarray(bases, dtype=np.float64),
        )

    def version(self):
        """Return the engine version as one compact line."""
        lines = (line.strip() for line in str(self._context.Basic.Version()).splitlines())
        return "; ".join(line for line in lines if line)
