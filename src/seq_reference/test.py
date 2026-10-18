"""Evaluate sequential neutral and RegControl references and write daily and hourly metrics."""

import pathlib
import sys

from common.cli import (
    ArgumentParser,
    beta_value,
    dataset_name,
    display_path,
    positive_integer,
    result_directory,
)
from common.evaluation import (
    run_day_test,
    run_native_day_test,
)
from common.outputs import (
    day_summary,
    write_day_test,
    write_hour_test,
)
from feeder.envs import DayEnv
from feeder.loads import dataset_feeder, read_days
from feeder.simulator import FeederSimulator

ROOT = pathlib.Path(__file__).resolve().parents[2]
CONTROLLERS = ("neutral", "regcontrol")


def parse_args():
    """Parse reference controllers, benchmark betas, split, limit, and overwrite policy."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=dataset_name, default="ieee13-random")
    parser.add_argument("--controller", choices=CONTROLLERS, nargs="+", required=True)
    parser.add_argument("--beta", type=beta_value, nargs="+", default=[1e-5])
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--days", type=positive_integer)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    """Evaluate requested controller-beta pairs on the same complete days.

    Neutral selects fixed neutral positions each hour. RegControl starts each
    native solve from the previous executed positions. Both routes reset each
    day at neutral, include initial movement in tap steps, and write daily and
    hourly CSVs. Every destination is checked before evaluation begins.
    """
    args = parse_args()
    try:
        simulator = FeederSimulator(dataset_feeder(args.dataset))
        selected_days = read_days(args.dataset, args.split, simulator.load_names, simulator.feeder)
        if args.days is not None:
            selected_days = selected_days[: args.days]
        base = result_directory(ROOT, args.dataset, "seq_reference")
        outputs = {}
        for controller in args.controller:
            for beta in args.beta:
                controller_dir = (base / f"{controller}_b{str(beta)}").resolve()
                if controller_dir.parent != base:
                    raise ValueError(f"unsafe controller directory: {controller_dir}")
                output = controller_dir / f"{args.split}.csv"
                hour_output = controller_dir / f"{args.split}_hourly.csv"
                outputs[(controller, beta)] = (output, hour_output)
                if not args.overwrite:
                    for destination in (output, hour_output):
                        if destination.exists():
                            raise ValueError(
                                f"{display_path(destination, ROOT)} exists; "
                                "use --overwrite to replace it"
                            )
    except (FileNotFoundError, ValueError) as error:
        print(error, file=sys.stderr)
        return 2

    print(f"dataset: data/scenarios/{args.dataset}")
    print(f"engine: DSS-Extensions {simulator.engine_version}")
    print(f"days: {len(selected_days)}")
    results = {(controller, beta): ([], []) for controller in args.controller for beta in args.beta}
    for beta in args.beta:
        env = DayEnv(simulator)
        if ("neutral", beta) in results:
            results[("neutral", beta)] = run_day_test(
                env, lambda _state: simulator.neutral_positions, selected_days, beta
            )
        if ("regcontrol", beta) in results:
            results[("regcontrol", beta)] = run_native_day_test(env, selected_days, beta)

    for controller in args.controller:
        for beta in args.beta:
            output, hour_output = outputs[(controller, beta)]
            rows, hour_rows = results[(controller, beta)]
            write_day_test(rows, output)
            write_hour_test(hour_rows, hour_output, simulator.action_count)
            print(f"wrote {display_path(output, ROOT)}")
            print(f"wrote {display_path(hour_output, ROOT)}")
            print(f"{controller}, beta {beta}: {day_summary(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
