"""Train sequential DDPG on complete days and write its run record, log, and checkpoints."""

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
from common.observations import day_observation, day_observation_size
from common.rewards import voltage_band_switching_reward
from common.runs import make_run_record, write_run_record
from common.sampling import sampling_rng, shuffled_passes
from ddpg.ddpg_vvo import Agent, save_models
from feeder.envs import DayEnv
from feeder.loads import dataset_feeder, read_days
from feeder.simulator import FeederSimulator
from feeder.taps import PHYSICAL_RANGE, project

ROOT = pathlib.Path(__file__).resolve().parents[2]
GAMMA = 1.0  # One complete day is the finite horizon.
WARMUP_DAYS = 20  # Uniform physical actions start the run.
EXPLORE_MIX = 0.1  # Later decisions keep this uniform-action share.
NOISE_SIGMA = 0.4  # Gaussian noise starts at this raw actor scale.
NOISE_FLOOR = 0.15  # Gaussian noise stays above this lower limit.
NOISE_DECAY = 0.99977  # Gaussian noise decays once per day.
CRITIC_WEIGHT_DECAY = 0.0  # The corrected critic has no weight decay.
ALPHA = 1e-4  # This value is the actor learning rate.
BETA = 1e-3  # This value is the critic learning rate.
TAU = 1e-3  # This fraction controls each Polyak target update.
BATCH_SIZE = 64  # Each learning update uses this many replay samples.
LAYER_SIZES = (400, 300)  # These values set the hidden-layer widths.
CHECKPOINT_EVERY = 200  # The pilot saves after this many days.
REWARDS = {
    "voltage_band_switching": (
        voltage_band_switching_reward,
        {"beta": 1e-5, "voltage_min": 0.95, "voltage_max": 1.05, "scale": 1000.0},
    )
}


def parse_args():
    """Parse training days, reward beta, benchmark beta, run, and overwrite policy."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=dataset_name, default="ieee13-random")
    parser.add_argument("--seed", type=seed_value, default=0)
    parser.add_argument("--days", type=positive_integer, default=1000)
    parser.add_argument(
        "--beta", type=beta_value, default=REWARDS["voltage_band_switching"][1]["beta"]
    )
    parser.add_argument("--benchmark-beta", type=beta_value, default=None)
    parser.add_argument("--reward", choices=tuple(REWARDS), default="voltage_band_switching")
    parser.add_argument("--run", type=run_name)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def train(
    agent,
    env,
    day_stream,
    days,
    rng,
    train_csv,
    checkpoint_dir,
    reward_fn,
    reward_parameters,
    observation_fn=day_observation,
    explore_mix=EXPLORE_MIX,
):
    """Train DDPG for at most ``days`` complete 24-decision episodes.

    Each day resets taps to neutral, then advances loads hourly while executed
    positions persist. The first 20 days use uniform physical proposals. Later
    interactions use a uniform proposal with probability ``explore_mix`` and a
    noisy actor proposal otherwise. Replay stores
    executed p.u. ratios, including terminal transitions at hour 23. With the
    configured gamma of one, nonterminal targets bootstrap across the finite day.

    The function attempts one learning update per interaction, decays Gaussian
    noise once per day, writes and flushes one CSV row per completed day, saves
    periodic checkpoints, and returns interaction and update counts.
    """
    lo, hi = PHYSICAL_RANGE
    updates = 0
    interactions = 0
    train_csv.parent.mkdir(parents=True, exist_ok=True)
    with train_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("episode", "day", "steps", "score", "beta"))
        for index, day in enumerate(itertools.islice(day_stream, days)):
            observation = observation_fn(env.reset(day))
            agent.noise.decay_sigma()
            score = 0.0
            steps = 0
            done = False
            while not done:
                if index < WARMUP_DAYS or rng.random() < explore_mix:
                    proposed = rng.uniform(lo, hi, size=env.action_count)
                else:
                    proposed, _noise = agent.choose_action(observation)
                positions, _requested_ratios = project(proposed)
                next_state, outcome, done = env.step(positions)
                next_observation = observation_fn(next_state)
                reward = reward_fn(outcome.result.voltages, outcome.tap_steps, **reward_parameters)
                agent.remember(
                    observation,
                    outcome.result.tap_ratios,
                    reward,
                    next_observation,
                    done,
                )
                updates += agent.learn() is not None
                interactions += 1
                observation = next_observation
                score += reward
                steps += 1

            episode = index + 1
            writer.writerow((episode, day.id, steps, round(score, 6), reward_parameters["beta"]))
            handle.flush()
            if episode % CHECKPOINT_EVERY == 0 or episode == days:
                save_models(agent, checkpoint_dir)
                print(f"episode {episode}: day {day.id} score {score:.6f}", flush=True)
    return interactions, updates


def main():
    """Validate a fresh run, record reward and benchmark settings, and train.

    Reward beta weights tap movement during learning. Benchmark beta is recorded
    separately for fixed evaluation metrics and may differ when an explicit run
    name is supplied. NumPy and PyTorch seeds control their respective training
    streams without promising cross-platform numerical identity.
    """
    args = parse_args()
    reward_fn, defaults = REWARDS[args.reward]
    reward_parameters = defaults.copy()
    reward_parameters["beta"] = args.beta
    benchmark_beta = args.beta if args.benchmark_beta is None else args.benchmark_beta
    selected_run = args.run if args.run is not None else f"b{str(args.beta)}_seed{args.seed}"
    try:
        if args.run is None and (
            args.reward != "voltage_band_switching" or benchmark_beta != args.beta
        ):
            raise ValueError("a non-default reward or benchmark beta needs an explicit --run")
        selected_run = run_name(selected_run)
        simulator = FeederSimulator(dataset_feeder(args.dataset))
        days = read_days(args.dataset, "train", simulator.load_names, simulator.feeder)
        run_dir = result_directory(ROOT, args.dataset, "seq_ddpg", selected_run)
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
    rng = np.random.default_rng(args.seed)
    print(f"dataset: data/scenarios/{args.dataset}")
    print(f"engine: DSS-Extensions {simulator.engine_version}")
    print(f"reward beta: {args.beta}")
    print(f"benchmark beta: {benchmark_beta}")
    env = DayEnv(simulator)
    lo, hi = PHYSICAL_RANGE
    agent = Agent(
        action_low=lo,
        action_high=hi,
        alpha=ALPHA,
        beta=BETA,
        input_dims=day_observation_size(simulator),
        tau=TAU,
        gamma=GAMMA,
        n_actions=env.action_count,
        batch_size=BATCH_SIZE,
        layer1_size=LAYER_SIZES[0],
        layer2_size=LAYER_SIZES[1],
        critic_wd=CRITIC_WEIGHT_DECAY,
        noise_sigma=NOISE_SIGMA,
        noise_floor=NOISE_FLOOR,
        noise_decay=NOISE_DECAY,
    )
    stream = shuffled_passes(days, sampling_rng(args.seed))
    record = make_run_record(
        feeder=simulator.feeder,
        dataset=args.dataset,
        algorithm="seq_ddpg",
        observation_name="day_observation",
        observation_size=day_observation_size(simulator),
        action_count=env.action_count,
        action_low=lo,
        action_high=hi,
        reward_name=args.reward,
        reward_parameters=reward_parameters,
        layer_sizes=LAYER_SIZES,
        benchmark_beta=benchmark_beta,
    )
    write_run_record(run_dir, record)
    interactions, updates = train(
        agent,
        env,
        stream,
        args.days,
        rng,
        run_dir / "train.csv",
        run_dir / "checkpoints",
        reward_fn=reward_fn,
        reward_parameters=reward_parameters,
    )
    print(f"interactions {interactions} learning_updates {updates}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
