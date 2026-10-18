"""Run static and complete-day evaluation under fixed operational metrics.

The protocols apply controllers to physical environments and create result rows.
They do not use or reconstruct the reward selected during training.
"""

from common.metrics import voltage_metrics
from feeder.loads import scenario_loads
from feeder.simulator import SimulationError


def static_row(scenario, result):
    """Build one scenario row with executed positions and operational metrics."""
    measured = voltage_metrics(result.voltages, converged=result.converged)
    row = {"scenario": scenario.id}
    row.update(
        (f"k{index}", int(position)) for index, position in enumerate(result.tap_positions, start=1)
    )
    row.update(
        vmin=measured["vmin"],
        vmax=measured["vmax"],
        cost=measured["cost"],
        viol_cnt=measured["violation_count"],
        feasible=int(measured["feasible"]),
    )
    return row


def run_static_test(env, controller, scenarios):
    """Evaluate one decision on each supplied scenario in order.

    Every scenario receives a fresh neutral reset. ``controller`` maps that
    physical reset state to tap positions. The returned rows preserve scenario
    order, which permits paired controller comparisons on identical inputs.
    """
    rows = []
    for scenario in scenarios:
        state = env.reset(scenario)
        _next_state, outcome, _done = env.step(controller(state))
        rows.append(static_row(scenario, outcome.result))
    return rows


def run_native_static_test(simulator, scenarios):
    """Select native RegControl taps, then score a common fixed-action solve.

    Native selection starts from the compiled case for every independent
    scenario. A second solve disables controls and applies the selected positions,
    so static native results share the fixed-tap scoring route used elsewhere.
    Either nonconverged solve stops evaluation.
    """
    rows = []
    for scenario in scenarios:
        kw, kvar = scenario_loads(scenario, simulator)
        native = simulator.solve_native_from_case(kw, kvar)
        if not native.converged:
            raise SimulationError(
                f"native selection solve did not converge for scenario {scenario.id}"
            )
        result = simulator.solve(kw, kvar, native.tap_positions)
        if not result.converged:
            raise SimulationError(f"scoring solve did not converge for scenario {scenario.id}")
        rows.append(static_row(scenario, result))
    return rows


def day_row(day, outcomes, beta):
    """Aggregate 24 hourly outcomes into one benchmark row.

    Voltage cost and tap-step distance are sums across hours. Daily cost adds
    ``beta`` times total steps. ``feasible_hours`` counts hours whose complete
    monitored voltage set is feasible; it is not a voltage-violation count.
    """
    voltage_cost = float(
        sum(
            voltage_metrics(outcome.result.voltages, converged=outcome.result.converged)["cost"]
            for outcome in outcomes
        )
    )
    steps = int(sum(outcome.tap_steps for outcome in outcomes))
    return {
        "day": day.id,
        "beta": beta,
        "voltage_cost": voltage_cost,
        "tap_steps": steps,
        "daily_cost": float(voltage_cost + beta * steps),
        "feasible_hours": int(
            sum(
                voltage_metrics(outcome.result.voltages, converged=outcome.result.converged)[
                    "feasible"
                ]
                for outcome in outcomes
            )
        ),
    }


def hour_row(day, outcome, beta):
    """Assess one executed hourly action under the benchmark beta.

    The row records positions before and after the action, summed movement,
    physical voltage metrics, and ``voltage_cost + beta * tap_steps``.
    """
    previous = outcome.previous_tap_positions
    positions = outcome.tap_positions
    result = outcome.result
    measured = voltage_metrics(result.voltages, converged=result.converged)
    row = {"day": day.id, "hour": outcome.hour, "beta": beta}
    row.update((f"k{index}_before", int(position)) for index, position in enumerate(previous, 1))
    row.update((f"k{index}", int(position)) for index, position in enumerate(positions, 1))
    row.update(
        tap_steps=outcome.tap_steps,
        voltage_cost=measured["cost"],
        switching_cost=float(beta * outcome.tap_steps),
        hourly_cost=float(measured["cost"] + beta * outcome.tap_steps),
        vmin=measured["vmin"],
        vmax=measured["vmax"],
        viol_cnt=measured["violation_count"],
        feasible=int(measured["feasible"]),
        converged=int(result.converged),
    )
    return row


def run_day_test(env, controller, days, beta):
    """Evaluate one controller for 24 persistent-tap decisions per day.

    Each day resets at neutral. The controller receives the prepared current-hour
    state, and the environment advances loads after executing its action. Daily
    and hourly rows preserve supplied day order for paired comparisons.
    """
    rows, hour_rows = [], []
    for day in days:
        state = env.reset(day)
        outcomes = []
        done = False
        while not done:
            state, outcome, done = env.step(controller(state))
            outcomes.append(outcome)
            hour_rows.append(hour_row(day, outcome, beta))
        rows.append(day_row(day, outcomes, beta))
    return rows, hour_rows


def run_native_day_test(env, days, beta):
    """Evaluate sequential RegControl directly from each prior executed position.

    Unlike the static native protocol, each native solve is both the controller
    selection and the scored physical result. Taps persist within a day and reset
    to neutral between days.
    """
    rows, hour_rows = [], []
    for day in days:
        env.reset(day)
        outcomes = []
        done = False
        while not done:
            _state, outcome, done = env.step_native()
            outcomes.append(outcome)
            hour_rows.append(hour_row(day, outcome, beta))
        rows.append(day_row(day, outcomes, beta))
    return rows, hour_rows
