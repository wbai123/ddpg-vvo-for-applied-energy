"""Read named multiplier datasets and produce ordered physical load inputs.

Dataset columns follow the simulator's fixed compiled load order. Multipliers
become absolute kW and kvar only at the simulator boundary.
"""

import csv
import dataclasses
import json
import pathlib
import re

import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SCENARIOS_ROOT = _REPO_ROOT / "data/scenarios"
SPLITS = ("train", "validation", "test")
_DATASET_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
HOURS = 24


@dataclasses.dataclass(frozen=True)
class Scenario:
    """Hold one hourly multiplier row and its optional day coordinates."""

    id: str
    factors: tuple
    day_id: str | None = None
    hour: int | None = None


@dataclasses.dataclass(frozen=True)
class Day:
    """Hold 24 chronological scenarios with one day identifier."""

    id: str
    hours: tuple


def dataset_dir(name):
    """Return the scenario directory for a path-safe dataset name."""
    if not isinstance(name, str) or _DATASET_NAME.fullmatch(name) is None:
        raise ValueError(f"invalid dataset name: {name!r}")
    return SCENARIOS_ROOT / name


def scenario_loads(scenario, simulator):
    """Scale nominal loads into absolute kW and kvar in compiled load order.

    Each factor scales both nominal active and reactive demand for its matching
    load. Returned float64 arrays are suitable for a simulator solve.
    """
    try:
        factors = np.asarray(scenario.factors, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"scenario {scenario.id} must contain valid load factors") from error
    count = len(simulator.load_names)
    if factors.ndim != 1 or factors.shape != (count,) or not np.isfinite(factors).all():
        raise ValueError(f"scenario {scenario.id} must contain {count} finite factors")
    return factors * simulator.nominal_kw, factors * simulator.nominal_kvar


def uniform_scenario(scale, count):
    """Build a static scenario that applies one multiplier to every load."""
    value = float(scale)
    return Scenario(id=f"g{value:.2f}", factors=tuple([value] * count))


def dataset_feeder(dataset):
    """Read the feeder identity recorded in named dataset metadata."""
    path = dataset_dir(dataset) / "generation.json"
    if not path.is_file():
        raise FileNotFoundError(f"{_display_path(path)} is missing")
    with path.open() as handle:
        feeder = json.load(handle).get("feeder")
    if not isinstance(feeder, str):
        raise ValueError(f"{_display_path(path)}: feeder is missing or invalid")
    return feeder


def _display_path(path):
    try:
        return path.resolve().relative_to(_REPO_ROOT)
    except ValueError:
        return path


def read_split(path, load_names):
    """Read one hourly split while enforcing its exact load-column order.

    Rows remain in file order. Day completeness and chronology are checked when
    the rows are grouped by :func:`read_days`.
    """
    source = pathlib.Path(path)
    split = source.stem
    if split not in SPLITS:
        raise ValueError(f"invalid split: {split!r}; expected one of {SPLITS}")
    display = _display_path(source)
    if not source.is_file():
        raise FileNotFoundError(
            f"{display} is missing; run uv run src/datasets/generate.py --dataset <name>"
        )
    expected = ("day_id", "hour", *load_names)
    scenarios = []
    with source.open(newline="") as handle:
        reader = csv.reader(handle)
        header = tuple(next(reader, ()))
        if header != expected:
            raise ValueError(f"{display}: invalid header; expected {expected}, found {header}")
        for line_number, row in enumerate(reader, start=2):
            if len(row) != len(expected):
                raise ValueError(
                    f"{display}:{line_number}: expected {len(expected)} fields, found {len(row)}"
                )
            day_id = row[0]
            try:
                hour = int(row[1])
            except ValueError as error:
                raise ValueError(f"{display}:{line_number}: hour must be an integer") from error
            try:
                factors = tuple(float(item) for item in row[2:])
            except ValueError as error:
                raise ValueError(
                    f"{display}:{line_number}: multipliers must be finite floats"
                ) from error
            if not np.isfinite(factors).all():
                raise ValueError(f"{display}:{line_number}: multipliers must be finite floats")
            scenarios.append(Scenario(f"{day_id}-h{hour:02d}", factors, day_id, hour))
    if not scenarios:
        raise ValueError(f"{display}: split holds no hourly rows")
    return scenarios


def read_days(dataset, split, load_names, feeder=None):
    """Read a named split as complete chronological days.

    When ``feeder`` is supplied, the dataset metadata must name that feeder.
    """
    if split not in SPLITS:
        raise ValueError(f"invalid split: {split!r}; expected one of {SPLITS}")
    if feeder is not None:
        recorded = dataset_feeder(dataset)
        if recorded != feeder:
            raise ValueError(f"dataset {dataset} names feeder {recorded}, expected {feeder}")
    return group_days(read_split(dataset_dir(dataset) / f"{split}.csv", load_names))


def flatten_hours(days):
    """Flatten complete days without changing their day and hour order."""
    return [scenario for day in days for scenario in day.hours]


def group_days(scenarios):
    """Group consecutive rows into unique days containing hours 0 through 23."""
    days = []
    seen = set()
    current_id = None
    current = []

    def finish():
        if not current:
            return
        day = Day(current_id, tuple(current))
        validate_day(day)
        days.append(day)
        seen.add(current_id)

    for scenario in scenarios:
        if scenario.day_id is None or scenario.hour is None:
            raise ValueError(f"scenario {scenario.id} has no day_id or hour")
        if scenario.day_id != current_id:
            finish()
            if scenario.day_id in seen:
                raise ValueError(f"day_id {scenario.day_id} appears in separate runs")
            current_id = scenario.day_id
            current = []
        current.append(scenario)
    finish()
    return days


def validate_day(day):
    """Require 24 ordered hours whose recorded day identifier matches the day."""
    ordered = all(scenario.hour == hour for hour, scenario in enumerate(day.hours))
    if len(day.hours) != HOURS or not ordered:
        raise ValueError(f"day {day.id} does not hold hours 0..23 in order")
    if any(scenario.day_id != day.id for scenario in day.hours):
        raise ValueError(f"day {day.id} contains a different day_id")
