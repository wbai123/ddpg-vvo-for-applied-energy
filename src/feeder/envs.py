"""Progress one-shot and complete-day physical control episodes.

The environments manage loads, tap persistence, state timing, and convergence
failures. They return physical state and outcomes; callers choose observation
encodings, learning rewards, and evaluation metrics.
"""

import dataclasses

import numpy as np

from feeder.loads import HOURS, scenario_loads, validate_day
from feeder.simulator import SimulationError, SolveResult
from feeder.taps import tap_steps


@dataclasses.dataclass(frozen=True)
class PhysicalState:
    """Describe the physical inputs available before a control decision.

    Voltages and tap positions belong to ``scenario_id`` at the recorded hour.
    Arrays are independent of later environment transitions.
    """

    voltages: np.ndarray
    tap_positions: np.ndarray
    scenario_id: str
    day_id: str | None
    hour: int | None


@dataclasses.dataclass(frozen=True)
class HourOutcome:
    """Describe one executed action and its physical power-flow result.

    ``tap_steps`` is the summed position distance from the previous held taps.
    The result uses the loads identified by ``scenario_id`` and ``hour``.
    """

    hour: int | None
    scenario_id: str
    previous_tap_positions: np.ndarray
    tap_positions: np.ndarray
    tap_steps: int
    result: SolveResult


class OneShotEnv:
    """Make one independent tap decision after a neutral fixed-tap reset.

    Reset and action use the same scenario loads. A successful step terminates
    the episode and measures movement from neutral. Any failed reset or step
    leaves the environment inactive until a later successful reset.
    """

    def __init__(self, simulator):
        """Bind one physical simulator without choosing an observation or reward."""
        self._simulator = simulator
        self.action_count = simulator.action_count
        self._scenario = None
        self._loads = None
        self._active = False

    def _state(self, result):
        return PhysicalState(
            result.voltages,
            result.tap_positions.copy(),
            self._scenario.id,
            self._scenario.day_id,
            self._scenario.hour,
        )

    def reset(self, scenario):
        """Solve the scenario at neutral taps and activate one decision.

        The returned state contains the scenario's neutral-tap voltages. A
        nonconverged reset raises ``SimulationError`` and leaves no active episode.
        """
        self._active = False
        kw, kvar = scenario_loads(scenario, self._simulator)
        result = self._simulator.solve(kw, kvar, self._simulator.neutral_positions)
        if not result.converged:
            raise SimulationError(f"reset solve did not converge for scenario {scenario.id}")
        self._scenario = scenario
        self._loads = (kw, kvar)
        self._active = True
        return self._state(result)

    def step(self, tap_positions):
        """Execute one tap-position vector and terminate the active episode.

        Returns the executed state, its outcome, and ``True``. The outcome uses
        the reset scenario loads and counts movement from neutral. A failed solve
        invalidates the episode while preserving simulator state for diagnosis.
        """
        if not self._active:
            raise RuntimeError("step() requires reset(scenario) first")
        self._active = False
        result = self._simulator.solve(*self._loads, tap_positions)
        if not result.converged:
            raise SimulationError(f"action solve did not converge for scenario {self._scenario.id}")
        previous = self._simulator.neutral_positions.copy()
        outcome = HourOutcome(
            self._scenario.hour,
            self._scenario.id,
            previous,
            result.tap_positions.copy(),
            tap_steps(result.tap_positions, previous),
            result,
        )
        return self._state(result), outcome, True


class DayEnv:
    """Make 24 hourly decisions while loads advance and tap positions persist.

    A day starts at hour 0 with neutral taps. Each step scores the current hour,
    retains the executed taps, and, except after hour 23, solves the next hour at
    those taps to form the next decision state. A failed solve during reset or
    step invalidates the episode.
    """

    def __init__(self, simulator):
        """Bind one simulator and initialize held taps outside an active day."""
        self._simulator = simulator
        self.action_count = simulator.action_count
        self._tap_positions = simulator.neutral_positions.copy()
        self._hour = None
        self._day = None
        self._active = False

    @property
    def tap_positions(self):
        """Return a copy of the last successfully executed positions."""
        return self._tap_positions.copy()

    @property
    def hour(self):
        """Return the current decision hour, or ``None`` before the first reset."""
        return self._hour

    @property
    def day(self):
        """Return the current validated day, or ``None`` before the first reset."""
        return self._day

    def _state(self, result):
        return PhysicalState(
            result.voltages,
            self._tap_positions.copy(),
            self._day.hours[self._hour].id,
            self._day.id,
            self._hour,
        )

    def reset(self, day):
        """Validate a complete day and prepare hour 0 at neutral taps.

        The returned state represents hour-0 loads before its action. Day
        validation happens before the current episode is deactivated, so an
        invalid day raises ``ValueError`` and leaves any active episode
        untouched. After validation, reset leaves the environment inactive
        unless initialization succeeds. A nonconverged initialization solve
        raises ``SimulationError``.
        """
        validate_day(day)
        self._active = False
        neutral_taps = self._simulator.neutral_positions.copy()
        kw, kvar = scenario_loads(day.hours[0], self._simulator)
        result = self._simulator.solve(kw, kvar, neutral_taps)
        if not result.converged:
            raise SimulationError(f"reset solve did not converge for day {day.id} hour 0")
        self._day = day
        self._hour = 0
        self._tap_positions = neutral_taps
        self._active = True
        return self._state(result)

    def _finish(self, result, previous):
        """Record an executed hour, retain its taps, and prepare the next state.

        For hours 0--22, the next state uses the next hour's loads at the
        executed positions. For hour 23, it repeats the terminal outcome state.
        If next-state preparation fails, executed taps remain retained, the hour
        remains the action hour, and the episode stays inactive.
        """
        outcome = HourOutcome(
            self._hour,
            self._day.hours[self._hour].id,
            previous.copy(),
            result.tap_positions.copy(),
            tap_steps(result.tap_positions, previous),
            result,
        )
        self._tap_positions = result.tap_positions.copy()
        done = self._hour == HOURS - 1
        if done:
            next_state = self._state(result)
        else:
            next_hour = self._hour + 1
            scenario = self._day.hours[next_hour]
            kw, kvar = scenario_loads(scenario, self._simulator)
            prepared = self._simulator.solve(kw, kvar, self._tap_positions)
            if not prepared.converged:
                raise SimulationError(
                    f"next-observation solve did not converge for day {self._day.id} "
                    f"hour {next_hour}"
                )
            self._hour = next_hour
            next_state = self._state(prepared)
            self._active = True
        return next_state, outcome, done

    def step(self, tap_positions):
        """Execute fixed tap positions for the current hour and advance the day."""
        if not self._active:
            raise RuntimeError("step() requires an active day: reset(day) first")
        self._active = False
        previous = self._tap_positions.copy()
        scenario = self._day.hours[self._hour]
        kw, kvar = scenario_loads(scenario, self._simulator)
        result = self._simulator.solve(kw, kvar, tap_positions)
        if not result.converged:
            raise SimulationError(
                f"action solve did not converge for day {self._day.id} hour {self._hour}"
            )
        return self._finish(result, previous)

    def step_native(self):
        """Let RegControl act from held taps for the current hour and advance."""
        if not self._active:
            raise RuntimeError("step_native() requires an active day: reset(day) first")
        self._active = False
        previous = self._tap_positions.copy()
        scenario = self._day.hours[self._hour]
        kw, kvar = scenario_loads(scenario, self._simulator)
        result = self._simulator.solve_native(kw, kvar, self._tap_positions)
        if not result.converged:
            raise SimulationError(
                f"action solve did not converge for day {self._day.id} hour {self._hour}"
            )
        return self._finish(result, previous)
