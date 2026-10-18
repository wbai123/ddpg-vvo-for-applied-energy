"""Train one-shot DDPG from a named dataset and write its run record, log, and checkpoints."""

import csv
import itertools
import pathlib
import shutil
import sys

import numpy as np
import torch as T

from common.cli import (
    ArgumentParser,
    beta_value,
    dataset_name,
    display_path,
    positive_integer,
    result_directory,
    run_name,
    seed_value,
)
from common.observations import voltage_observation, voltage_observation_size
from common.rewards import voltage_band_reward
from common.runs import make_run_record, write_run_record
from common.sampling import sampling_rng, shuffled_passes
from ddpg.ddpg_vvo import Agent, save_models
from feeder.envs import OneShotEnv
from feeder.loads import dataset_feeder, flatten_hours, read_days
from feeder.simulator import FeederSimulator
from feeder.taps import PHYSICAL_RANGE, project

ROOT = pathlib.Path(__file__).resolve().parents[2]
MAX_STEPS = 1  # Each sampled context has one step.
GAMMA = 0.0  # The one-step episode uses no bootstrap term.
WARMUP = 1000  # Uniform physical actions start the run.
EXPLORE_MIX = 0.1  # Later episodes keep this random-action share.
NOISE_DECAY = 0.99977  # Gaussian noise decays once per episode.
CRITIC_WEIGHT_DECAY = 0.0  # The corrected critic has no weight decay.
ALPHA = 1e-4  # This value is the actor learning rate.
BETA = 1e-3  # This value is the critic learning rate.
TAU = 1e-3  # This fraction controls each Polyak target update.
BATCH_SIZE = 64  # Each learning update uses this many replay samples.
LAYER_SIZES = (400, 300)  # These values set the hidden-layer widths.
REWARDS = {
    "voltage_band": (
        voltage_band_reward,
        {"mad_weight": 0.0, "voltage_min": 0.95, "voltage_max": 1.05, "scale": 1000.0},
    )
}


def parse_args():
    """Parse the dataset, seed, episode budget, reward, run, and overwrite policy."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=dataset_name, default="ieee13-random")
    parser.add_argument("--seed", type=seed_value, default=0)
    parser.add_argument("--episodes", type=positive_integer, default=20000)
    parser.add_argument("--run", type=run_name)
    parser.add_argument("--reward", choices=tuple(REWARDS), default="voltage_band")
    parser.add_argument(
        "--mad-weight", type=beta_value, default=REWARDS["voltage_band"][1]["mad_weight"]
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def train(
    agent,
    env,
    scenarios,
    episodes,
    train_csv,
    checkpoint_dir,
    reward_fn,
    reward_parameters,
    observation_fn=voltage_observation,
):
    """Train DDPG for at most ``episodes`` independent one-step episodes.

    Every episode resets one scenario at neutral taps, projects a continuous
    proposal to integer positions, and stores the executed p.u. ratios in
    replay. The first 1,000 episodes use uniform physical proposals. Later
    episodes mix uniform proposals with noisy actor proposals. Learning is
    attempted once per interaction and begins after replay fills one batch.

    The function writes and flushes one CSV row per episode, saves both online
    networks periodically and at completion, and returns distinct counts of
    environment interactions and completed learning updates.
    """
    lo, hi = PHYSICAL_RANGE
    updates = 0
    interactions = 0
    train_csv.parent.mkdir(parents=True, exist_ok=True)
    with train_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("episode", "scenario", "steps", "score"))
        for index, scenario in enumerate(itertools.islice(scenarios, episodes)):
            observation = observation_fn(env.reset(scenario))
            agent.noise.decay_sigma()
            if index < WARMUP or np.random.random() < EXPLORE_MIX:
                proposed = np.random.uniform(lo, hi, size=env.action_count)
            else:
                proposed, _noise = agent.choose_action(observation)
            positions, _requested_ratios = project(proposed)
            next_state, outcome, done = env.step(positions)
            next_observation = observation_fn(next_state)
            reward = reward_fn(outcome.result.voltages, **reward_parameters)
            agent.remember(observation, outcome.result.tap_ratios, reward, next_observation, done)
            updates += agent.learn() is not None
            interactions += 1
            episode = index + 1
            writer.writerow((episode, scenario.id, MAX_STEPS, round(reward, 6)))
            handle.flush()
            if episode % 2000 == 0:
                save_models(agent, checkpoint_dir)
                print(f"episode {episode}: score {reward:.6f}", flush=True)
    save_models(agent, checkpoint_dir)
    return interactions, updates


def main():
    """Validate a fresh run, seed training streams, record settings, and train.

    NumPy controls exploration and replay sampling; PyTorch controls network
    initialization and learning operations. Fixed seeds support repeated runs
    in the pinned environment but do not promise identity on every platform.
    """
    args = parse_args()
    reward_fn, defaults = REWARDS[args.reward]
    reward_parameters = defaults.copy()
    reward_parameters["mad_weight"] = args.mad_weight
    selected_run = args.run if args.run is not None else f"seed{args.seed}"
    try:
        if args.run is None and (args.reward != "voltage_band" or args.mad_weight != 0.0):
            raise ValueError("a non-default reward needs an explicit --run")
        selected_run = run_name(selected_run)
        simulator = FeederSimulator(dataset_feeder(args.dataset))
        scenarios = flatten_hours(
            read_days(args.dataset, "train", simulator.load_names, simulator.feeder)
        )
        run_dir = result_directory(ROOT, args.dataset, "ddpg", selected_run)
        if run_dir.exists() and not args.overwrite:
            raise ValueError(
                f"{display_path(run_dir, ROOT)} exists; use --overwrite to replace the run"
            )
    except (FileNotFoundError, ValueError) as error:
        print(error, file=sys.stderr)
        return 2
    if args.overwrite and run_dir.exists():
        shutil.rmtree(run_dir)

    np.random.seed(args.seed)
    T.manual_seed(args.seed)
    print(f"dataset: data/scenarios/{args.dataset}")
    print(f"engine: DSS-Extensions {simulator.engine_version}")
    env = OneShotEnv(simulator)
    lo, hi = PHYSICAL_RANGE
    agent = Agent(
        action_low=lo,
        action_high=hi,
        alpha=ALPHA,
        beta=BETA,
        input_dims=voltage_observation_size(simulator),
        tau=TAU,
        gamma=GAMMA,
        n_actions=env.action_count,
        batch_size=BATCH_SIZE,
        layer1_size=LAYER_SIZES[0],
        layer2_size=LAYER_SIZES[1],
        critic_wd=CRITIC_WEIGHT_DECAY,
        noise_decay=NOISE_DECAY,
    )
    stream = shuffled_passes(scenarios, sampling_rng(args.seed))
    record = make_run_record(
        feeder=simulator.feeder,
        dataset=args.dataset,
        algorithm="ddpg",
        observation_name="voltage_observation",
        observation_size=voltage_observation_size(simulator),
        action_count=env.action_count,
        action_low=lo,
        action_high=hi,
        reward_name=args.reward,
        reward_parameters=reward_parameters,
        layer_sizes=LAYER_SIZES,
    )
    write_run_record(run_dir, record)
    interactions, updates = train(
        agent,
        env,
        stream,
        args.episodes,
        run_dir / "train.csv",
        run_dir / "checkpoints",
        reward_fn=reward_fn,
        reward_parameters=reward_parameters,
    )
    print(f"interactions {interactions} learning_updates {updates}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
