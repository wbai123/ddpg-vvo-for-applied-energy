"""Train sequential TD3 on complete days and record its recipe and update counts."""

import csv
import itertools
import json
import pathlib
import shutil
import sys

import numpy as np
import torch

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
from feeder.envs import DayEnv
from feeder.loads import dataset_feeder, read_days
from feeder.simulator import FeederSimulator
from feeder.taps import PHYSICAL_RANGE, project
from td3.td3_vvo import Agent, save_models

ROOT = pathlib.Path(__file__).resolve().parents[2]
SETTINGS = {
    "layer_sizes": (400, 300),
    "gamma": 1.0,
    "actor_learning_rate": 3e-4,
    "critic_learning_rate": 3e-4,
    "batch_size": 256,
    "replay_capacity": 1_000_000,
    "replay_sampling": "with_replacement",
    "tau": 0.005,
    "policy_delay": 2,
    "target_smoothing_sigma_normalized": 0.2,
    "target_smoothing_sigma_pu": 0.02,
    "target_smoothing_clip_normalized": 0.5,
    "target_smoothing_clip_pu": 0.05,
    "exploration_sigma_normalized": 0.1,
    "exploration_sigma_pu": 0.01,
    "initialization": "torch_default_linear",
    "checkpoint_support": "inference_only",
}
WARMUP_INTERACTIONS = 480  # Twenty complete days use uniform physical proposals.
LEARNING_START_INTERACTION = 481  # The first post-warmup transition trains both critics.
CRITIC_ITERATIONS_PER_INTERACTION = 1  # Each eligible interaction makes one critic update.
CHECKPOINT_EVERY = 200  # Save inference checkpoints at this day interval and at the end.
REWARDS = {
    "voltage_band_switching": (
        voltage_band_switching_reward,
        {"beta": 1e-5, "voltage_min": 0.95, "voltage_max": 1.05, "scale": 1000.0},
    )
}


def parse_args():
    """Parse the dataset, fixed-budget recipe, run name, and overwrite policy."""
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
    counts_path,
    reward_fn,
    reward_parameters,
    *,
    exploration_sigma,
    warmup_interactions,
    learning_start_interaction,
    observation_fn=day_observation,
):
    """Train TD3 over complete days and write durable progress and counts.

    Every day starts from neutral taps. Uniform physical proposals fill the
    warmup. Later proposals add Gaussian exploration once in physical p.u.
    units to deterministic actor ratios, clip to the physical bounds, and then
    project to executable taps. ``exploration_sigma`` is a number or a function
    of the one-based interaction number. Replay stores the executed ratios, not the
    proposal. The environment marks hour 23 terminal, so the learner removes
    its bootstrap term there. One critic iteration follows each eligible stored
    transition; actor and target updates follow the learner's policy delay.
    """
    low, high = PHYSICAL_RANGE
    sigma_at = exploration_sigma if callable(exploration_sigma) else lambda _: exploration_sigma
    interactions = 0
    train_csv.parent.mkdir(parents=True, exist_ok=True)
    with train_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("episode", "day", "steps", "score", "beta"))
        for index, day in enumerate(itertools.islice(day_stream, days)):
            observation = observation_fn(env.reset(day))
            score = 0.0
            steps = 0
            done = False
            while not done:
                interaction = interactions + 1
                if interaction <= warmup_interactions:
                    proposed = rng.uniform(low, high, size=env.action_count)
                else:
                    noise = rng.normal(0.0, sigma_at(interaction), size=env.action_count)
                    proposed = np.clip(agent.act(observation) + noise, low, high)
                positions, _requested_ratios = project(proposed)
                next_state, outcome, done = env.step(positions)
                next_observation = observation_fn(next_state)
                reward = reward_fn(
                    outcome.result.voltages,
                    outcome.tap_steps,
                    **reward_parameters,
                )
                agent.remember(
                    observation,
                    outcome.result.tap_ratios,
                    reward,
                    next_observation,
                    done,
                )
                if interaction >= learning_start_interaction:
                    agent.learn()
                interactions = interaction
                observation = next_observation
                score += reward
                steps += 1

            episode = index + 1
            writer.writerow((episode, day.id, steps, round(score, 6), reward_parameters["beta"]))
            handle.flush()
            if episode % CHECKPOINT_EVERY == 0 or episode == days:
                save_models(agent, checkpoint_dir)
                print(f"episode {episode}: day {day.id} score {score:.6f}", flush=True)
    counts = {
        "days": days,
        "interactions": interactions,
        "critic_iterations": agent.critic_iterations,
        "actor_updates": agent.actor_updates,
        "target_updates": agent.target_updates,
    }
    counts_path.write_text(json.dumps(counts, indent=2) + "\n")
    return counts


def _agent_settings(seed, training_days):
    """Build the saved recipe from the exact values supplied to training."""
    low, high = PHYSICAL_RANGE
    return {
        "seed": seed,
        "random_streams": {
            "day_order": [seed, 1],
            "learner": [seed, 2],
            "torch": seed,
        },
        **{name: value for name, value in SETTINGS.items() if name != "layer_sizes"},
        "warmup_interactions": WARMUP_INTERACTIONS,
        "warmup_days": WARMUP_INTERACTIONS // 24,
        "learning_start_interaction": LEARNING_START_INTERACTION,
        "critic_iterations_per_interaction": CRITIC_ITERATIONS_PER_INTERACTION,
        "action_map": {
            "midpoint": (low + high) / 2,
            "half_range": (high - low) / 2,
            "normalized_range": [-1.0, 1.0],
        },
        "training_days": training_days,
    }


def main():
    """Validate, record, and train one fresh sequential TD3 run.

    Reward beta weights tap movement in the training reward. Benchmark beta
    independently weights the reported operational cost. The training-order,
    learner, and PyTorch streams are recorded explicitly before any updates.
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
        run_dir = result_directory(ROOT, args.dataset, "seq_td3", selected_run)
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
    torch.manual_seed(args.seed)
    learner_rng = np.random.default_rng([args.seed, 2])
    print(f"dataset: data/scenarios/{args.dataset}")
    print(f"engine: DSS-Extensions {simulator.engine_version}")
    print(f"reward beta: {args.beta}")
    print(f"benchmark beta: {benchmark_beta}")
    env = DayEnv(simulator)
    low, high = PHYSICAL_RANGE
    agent = Agent(
        observation_size=day_observation_size(simulator),
        action_count=env.action_count,
        action_low=low,
        action_high=high,
        gamma=SETTINGS["gamma"],
        rng=learner_rng,
        actor_learning_rate=SETTINGS["actor_learning_rate"],
        critic_learning_rate=SETTINGS["critic_learning_rate"],
        batch_size=SETTINGS["batch_size"],
        replay_capacity=SETTINGS["replay_capacity"],
        tau=SETTINGS["tau"],
        policy_delay=SETTINGS["policy_delay"],
        target_smoothing_sigma=SETTINGS["target_smoothing_sigma_pu"],
        target_smoothing_clip=SETTINGS["target_smoothing_clip_pu"],
        layer_sizes=SETTINGS["layer_sizes"],
    )
    record = make_run_record(
        feeder=simulator.feeder,
        dataset=args.dataset,
        algorithm="seq_td3",
        observation_name="day_observation",
        observation_size=day_observation_size(simulator),
        action_count=env.action_count,
        action_low=low,
        action_high=high,
        reward_name=args.reward,
        reward_parameters=reward_parameters,
        layer_sizes=SETTINGS["layer_sizes"],
        benchmark_beta=benchmark_beta,
        agent_settings=_agent_settings(args.seed, args.days),
    )
    write_run_record(run_dir, record)
    stream = shuffled_passes(days, sampling_rng(args.seed))
    counts = train(
        agent,
        env,
        stream,
        args.days,
        learner_rng,
        run_dir / "train.csv",
        run_dir / "checkpoints",
        run_dir / "counts.json",
        reward_fn,
        reward_parameters,
        exploration_sigma=SETTINGS["exploration_sigma_pu"],
        warmup_interactions=WARMUP_INTERACTIONS,
        learning_start_interaction=LEARNING_START_INTERACTION,
    )
    print(
        "counts: "
        f"interactions {counts['interactions']}, "
        f"critic_iterations {counts['critic_iterations']}, "
        f"actor_updates {counts['actor_updates']}, "
        f"target_updates {counts['target_updates']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
