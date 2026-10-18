"""Evaluate a saved one-shot DDPG actor and write static physical metrics for a named split."""

import pathlib
import sys

from common.cli import (
    ArgumentParser,
    dataset_name,
    display_path,
    positive_integer,
    result_directory,
    run_name,
)
from common.evaluation import run_static_test
from common.observations import voltage_observation, voltage_observation_size
from common.outputs import summary, write_static_test
from ddpg.checkpoints import load_agent_for_run
from feeder.envs import OneShotEnv
from feeder.loads import dataset_feeder, flatten_hours, read_days
from feeder.simulator import FeederSimulator
from feeder.taps import PHYSICAL_RANGE, project

ROOT = pathlib.Path(__file__).resolve().parents[2]


def parse_args():
    """Parse the saved run, evaluation split, optional day limit, and overwrite policy."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=dataset_name, default="ieee13-random")
    parser.add_argument("--run", type=run_name, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--days", type=positive_integer)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    """Load a compatible actor, evaluate matched scenarios, and write one CSV.

    Each scenario starts from neutral taps. The deterministic actor proposes
    p.u. ratios, which are projected to executable positions before the fixed
    power flow. Evaluation uses common operational metrics, not training reward.
    Existing output is rejected unless ``--overwrite`` is present.
    """
    args = parse_args()
    try:
        simulator = FeederSimulator(dataset_feeder(args.dataset))
        selected_days = read_days(args.dataset, args.split, simulator.load_names, simulator.feeder)
        if args.days is not None:
            selected_days = selected_days[: args.days]
        run_dir = result_directory(ROOT, args.dataset, "ddpg", args.run)
        output = run_dir / f"{args.split}.csv"
        if output.exists() and not args.overwrite:
            raise ValueError(f"{display_path(output, ROOT)} exists; use --overwrite to replace it")
        env = OneShotEnv(simulator)
        agent, _task = load_agent_for_run(
            run_dir,
            feeder=simulator.feeder,
            dataset=args.dataset,
            algorithm="ddpg",
            observation_name="voltage_observation",
            observation_size=voltage_observation_size(simulator),
            action_count=env.action_count,
            action_bounds=PHYSICAL_RANGE,
        )
    except (FileNotFoundError, ValueError) as error:
        print(error, file=sys.stderr)
        return 2

    print(f"dataset: data/scenarios/{args.dataset}")
    print(f"engine: DSS-Extensions {simulator.engine_version}")
    print(f"days: {len(selected_days)}")

    def controller(state):
        positions, _ratios = project(agent.choose_action_deterministic(voltage_observation(state)))
        return positions

    rows = run_static_test(env, controller, flatten_hours(selected_days))
    write_static_test(rows, output, env.action_count)
    print(f"wrote {display_path(output, ROOT)}")
    print(summary(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
