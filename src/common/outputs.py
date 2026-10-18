"""Define evaluation CSV schemas, writers, aggregate summaries, and paired counts."""

import csv
import pathlib
import statistics

import numpy as np

DAY_COLUMNS = ("day", "beta", "voltage_cost", "tap_steps", "daily_cost", "feasible_hours")


def static_columns(action_count):
    """Return static result columns with positions in regulator order."""
    return (
        "scenario",
        *(f"k{index}" for index in range(1, action_count + 1)),
        "vmin",
        "vmax",
        "cost",
        "viol_cnt",
        "feasible",
    )


def hour_columns(action_count):
    """Return hourly columns with before and after positions in regulator order."""
    return (
        "day",
        "hour",
        "beta",
        *(f"k{index}_before" for index in range(1, action_count + 1)),
        *(f"k{index}" for index in range(1, action_count + 1)),
        "tap_steps",
        "voltage_cost",
        "switching_cost",
        "hourly_cost",
        "vmin",
        "vmax",
        "viol_cnt",
        "feasible",
        "converged",
    )


def _write(rows, path, columns):
    """Create parent directories and replace one CSV with the declared schema."""
    output = pathlib.Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_static_test(rows, path, action_count):
    """Write one ordered row per static scenario, replacing ``path``."""
    _write(rows, path, static_columns(action_count))


def write_day_test(rows, path):
    """Write one aggregate result row per day, replacing ``path``."""
    _write(rows, path, DAY_COLUMNS)


def write_hour_test(rows, path, action_count):
    """Write ordered hourly details for each day, replacing ``path``."""
    _write(rows, path, hour_columns(action_count))


def summary(rows):
    """Return static mean cost and feasible-scenario count for command output."""
    return (
        f"{len(rows)} scenarios: mean cost "
        f"{np.mean([row['cost'] for row in rows]):.6g}, "
        f"feasible {sum(row['feasible'] for row in rows)}/{len(rows)}"
    )


def day_summary(rows):
    """Return mean daily costs, movement, and feasible hours for command output."""
    return (
        f"{len(rows)} days: mean daily cost "
        f"{np.mean([row['daily_cost'] for row in rows]):.6g}, "
        f"mean voltage cost {np.mean([row['voltage_cost'] for row in rows]):.6g}, "
        f"mean tap steps {np.mean([row['tap_steps'] for row in rows]):.6g}, "
        f"mean feasible hours {np.mean([row['feasible_hours'] for row in rows]):.6g}"
    )


def static_summary(rows):
    """Average physical metrics across one controller's matched scenarios."""
    count = len(rows)
    return {
        "Mean hourly voltage cost": statistics.fmean(float(row["cost"]) for row in rows),
        "Feasible scenarios (%)": 100 * sum(int(row["feasible"]) for row in rows) / count,
        "Mean violating voltage measurements/scenario": statistics.fmean(
            float(row["viol_cnt"]) for row in rows
        ),
        "Scenarios": count,
    }


def daily_summary(rows, beta):
    """Average daily metrics and derive mean switching cost under ``beta``.

    Across-day means describe the selected evaluation sample. They do not
    measure variation across independently trained seeds.
    """
    count = len(rows)
    mean_steps = statistics.fmean(float(row["tap_steps"]) for row in rows)
    return {
        "Mean daily combined cost": statistics.fmean(float(row["daily_cost"]) for row in rows),
        "Mean daily voltage cost": statistics.fmean(float(row["voltage_cost"]) for row in rows),
        "Mean daily switching cost": beta * mean_steps,
        "Mean tap steps/day": mean_steps,
        "Feasible hours (%)": 100 * sum(int(row["feasible_hours"]) for row in rows) / (24 * count),
        "Days": count,
    }


def paired_counts(left, right, column):
    """Count exact lower, equal, and higher values on position-paired rows.

    Equality uses exact numeric comparison and is not a practical-equivalence
    tolerance. Callers must establish matching row identities and order.
    """
    left_values = np.asarray([float(row[column]) for row in left])
    right_values = np.asarray([float(row[column]) for row in right])
    if left_values.shape != right_values.shape:
        raise ValueError("paired result lengths differ")
    return (
        int(np.sum(left_values < right_values)),
        int(np.sum(left_values == right_values)),
        int(np.sum(left_values > right_values)),
    )
