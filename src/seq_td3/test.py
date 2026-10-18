"""Evaluate a saved sequential TD3 actor over complete persistent-tap days."""

import pathlib
import sys

from common.cli import (
    ArgumentParser,
    beta_value,
    dataset_name,
    display_path,
    positive_integer,
    result_directory,
    run_name,
)
from common.evaluation import run_day_test
from common.observations import day_observation, day_observation_size
from common.outputs import day_summary, write_day_test, write_hour_test
from feeder.envs import DayEnv
from feeder.loads import dataset_feeder, read_days
from feeder.simulator import FeederSimulator
from feeder.taps import PHYSICAL_RANGE, project
from td3.checkpoints import load_agent_for_run

ROOT = pathlib.Path(__file__).resolve().parents[2]


def parse_args():
    """Parse the saved run, split, day limit, benchmark override, and overwrite policy."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=dataset_name, default="ieee13-random")
    parser.add_argument("--run", type=run_name, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--days", type=positive_integer)
    parser.add_argument("--benchmark-beta", type=beta_value, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    """Load deterministic inference, evaluate complete days, and write paired CSVs.

    The actor proposes continuous physical ratios without exploration. Grid
    projection determines executed taps before each solve. The saved benchmark
    beta controls reported switching cost independently of the training reward.
    Both output files remain protected as a pair unless overwrite is explicit.
    """
    args = parse_args()
    try:
        run_dir = result_directory(ROOT, args.dataset, "seq_td3", args.run)
        simulator = FeederSimulator(dataset_feeder(args.dataset))
        selected_days = read_days(args.dataset, args.split, simulator.load_names, simulator.feeder)
        if args.days is not None:
            selected_days = selected_days[: args.days]
        suffix = "" if args.benchmark_beta is None else f"_b{str(args.benchmark_beta)}"
        output = run_dir / f"{args.split}{suffix}.csv"
        hour_output = run_dir / f"{args.split}{suffix}_hourly.csv"
        if not args.overwrite:
            for destination in (output, hour_output):
                if destination.exists():
                    raise ValueError(
                        f"{display_path(destination, ROOT)} exists; use --overwrite to replace it"
                    )
        env = DayEnv(simulator)
        agent, task = load_agent_for_run(
            run_dir,
            feeder=simulator.feeder,
            dataset=args.dataset,
            algorithm="seq_td3",
            observation_name="day_observation",
            observation_size=day_observation_size(simulator),
            action_count=env.action_count,
            action_bounds=PHYSICAL_RANGE,
        )
        beta = task["evaluation"]["beta"] if args.benchmark_beta is None else args.benchmark_beta
    except (FileNotFoundError, ValueError) as error:
        print(error, file=sys.stderr)
        return 2

    print(f"dataset: data/scenarios/{args.dataset}")
    print(f"engine: DSS-Extensions {simulator.engine_version}")
    print(f"days: {len(selected_days)}")
    print(f"benchmark beta: {beta}")

    def controller(state):
        positions, _ratios = project(agent.act(day_observation(state)))
        return positions

    rows, hour_rows = run_day_test(env, controller, selected_days, beta)
    write_day_test(rows, output)
    write_hour_test(hour_rows, hour_output, env.action_count)
    print(f"wrote {display_path(output, ROOT)}")
    print(f"wrote {display_path(hour_output, ROOT)}")
    print(day_summary(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
