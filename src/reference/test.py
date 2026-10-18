"""Evaluate neutral taps or native RegControl and write static metrics for a named split."""

import pathlib
import sys

from common.cli import (
    ArgumentParser,
    dataset_name,
    display_path,
    positive_integer,
    result_directory,
)
from common.evaluation import run_native_static_test, run_static_test
from common.outputs import summary, write_static_test
from feeder.envs import OneShotEnv
from feeder.loads import dataset_feeder, flatten_hours, read_days
from feeder.simulator import FeederSimulator

ROOT = pathlib.Path(__file__).resolve().parents[2]


def parse_args():
    """Parse the reference controller, split, optional day limit, and overwrite policy."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=dataset_name, default="ieee13-random")
    parser.add_argument("--controller", choices=("neutral", "regcontrol"), required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--days", type=positive_integer)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    """Evaluate one static reference on the selected scenarios and write one CSV.

    Neutral evaluation uses the one-shot environment at fixed neutral positions.
    RegControl recompiles the case for native tap selection, then scores those
    positions with a common fixed-action solve. Both routes use the same ordered
    scenarios and operational metrics. Existing output is protected by default.
    """
    args = parse_args()
    try:
        simulator = FeederSimulator(dataset_feeder(args.dataset))
        selected_days = read_days(args.dataset, args.split, simulator.load_names, simulator.feeder)
        if args.days is not None:
            selected_days = selected_days[: args.days]
        base = result_directory(ROOT, args.dataset, "reference")
        controller_dir = (base / args.controller).resolve()
        if controller_dir.parent != base:
            raise ValueError(f"unsafe controller directory: {controller_dir}")
        output = controller_dir / f"{args.split}.csv"
        if output.exists() and not args.overwrite:
            raise ValueError(f"{display_path(output, ROOT)} exists; use --overwrite to replace it")
    except (FileNotFoundError, ValueError) as error:
        print(error, file=sys.stderr)
        return 2

    print(f"dataset: data/scenarios/{args.dataset}")
    print(f"engine: DSS-Extensions {simulator.engine_version}")
    print(f"days: {len(selected_days)}")
    scenarios = flatten_hours(selected_days)
    if args.controller == "neutral":
        rows = run_static_test(
            OneShotEnv(simulator), lambda _obs: simulator.neutral_positions, scenarios
        )
    else:
        rows = run_native_static_test(simulator, scenarios)
    write_static_test(rows, output, simulator.action_count)
    print(f"wrote {display_path(output, ROOT)}")
    print(summary(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
