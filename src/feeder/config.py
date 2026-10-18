"""Read feeder case paths, controlled regulators, and voltage conventions.

The simulator uses this configuration to bind each named experiment to its
compiled equipment and feeder-specific voltage measurements.
"""

import dataclasses
import pathlib
import tomllib

ROOT = pathlib.Path(__file__).resolve().parents[2]
FEEDERS = ("ieee13", "ieee37std", "ieee37mod")


@dataclasses.dataclass(frozen=True)
class FeederConfig:
    """Describe one supported feeder's physical and measurement configuration."""

    name: str
    case_path: pathlib.Path
    regulator_names: tuple[str, ...]
    windings: tuple[int, ...]
    voltage_convention: str
    voltage_selection: str
    native_control_iterations: int | None
    loads: tuple[str, ...] | None


def read_config(name):
    """Return a validated configuration for one of the three named feeders.

    Validation prevents load or regulator ordering changes and rejects voltage
    conventions that the simulator does not implement.
    """
    if name not in FEEDERS:
        raise ValueError(f"unknown feeder {name!r}; expected one of {FEEDERS}")
    directory = ROOT / "data" / name
    with (directory / "feeder.toml").open("rb") as handle:
        data = tomllib.load(handle)
    case = directory / data["case"]
    if not case.is_file() or case.parent != directory:
        raise FileNotFoundError(f"feeder case is missing: {case}")
    regulators = data["regulators"]
    names = tuple(item["name"] for item in regulators)
    windings = tuple(item["winding"] for item in regulators)
    if not names or len(set(name.lower() for name in names)) != len(names):
        raise ValueError(f"{name}: regulator names must be distinct")
    if any(winding != 2 for winding in windings):
        raise ValueError(f"{name}: only controlled winding 2 is supported")
    voltage = data["voltage"]
    expected = (
        ("line-ground", "initial-live-nodes")
        if name == "ieee13"
        else ("line-line", "three-phase-pairs-except-sourcebus")
    )
    convention = voltage["convention"]
    selection = voltage["selection"]
    if (convention, selection) != expected:
        raise ValueError(f"{name}: unsupported voltage measurement convention")
    limit = data.get("native_control_iterations")
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError(f"{name}: native control iteration limit must be positive")
    declared_loads = data.get("loads")
    if declared_loads is not None:
        if (
            not isinstance(declared_loads, list)
            or not declared_loads
            or any(not isinstance(item, str) or not item for item in declared_loads)
            or len(set(declared_loads)) != len(declared_loads)
        ):
            raise ValueError(f"{name}: declared load names must be distinct strings")
        declared_loads = tuple(declared_loads)
    return FeederConfig(name, case, names, windings, convention, selection, limit, declared_loads)
