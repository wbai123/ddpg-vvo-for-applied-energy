"""Reconstruct TD3 inference from saved records and inference-only checkpoints.

Evaluation validates task identity, the complete saved learner recipe, action
semantics, checkpoint keys, and tensor shapes before result output is written.
Optimizer state is deliberately absent, so checkpoints do not support training
resumption.
"""

import numpy as np
import torch

from common.runs import read_run_record
from td3.td3_vvo import Agent


def _checked_state(path, expected):
    """Read a state dictionary whose keys and tensor shapes match a new network."""
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint is missing: {path}")
    try:
        saved = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError(f"cannot read checkpoint {path}: {error}") from error
    if not isinstance(saved, dict) or set(saved) != set(expected):
        raise ValueError(f"checkpoint keys disagree with run record: {path}")
    for key, template in expected.items():
        if not isinstance(saved[key], torch.Tensor) or saved[key].shape != template.shape:
            raise ValueError(f"checkpoint tensor {key} disagrees with run record: {path}")
    return saved


def load_agent_for_run(
    run_dir,
    *,
    feeder,
    dataset,
    algorithm,
    observation_name,
    observation_size,
    action_count,
    action_bounds,
):
    """Return a deterministic TD3 agent reconstructed from one compatible run."""
    if algorithm != "seq_td3":
        raise ValueError(f"TD3 checkpoints require algorithm seq_td3, not {algorithm}")
    record = read_run_record(
        run_dir,
        feeder=feeder,
        dataset=dataset,
        algorithm=algorithm,
        observation_name=observation_name,
        observation_size=observation_size,
        action_count=action_count,
    )
    task = record["task"]
    action = task["action"]
    if (action["low"], action["high"]) != tuple(action_bounds):
        raise ValueError(f"run action bounds disagree with supported feeder bounds: {run_dir}")
    if task["evaluation"].get("beta") is None:
        raise ValueError(f"benchmark beta is missing from run record: {run_dir}")
    settings = record["agent"]["settings"]
    agent = Agent(
        observation_size=observation_size,
        action_count=action_count,
        action_low=action["low"],
        action_high=action["high"],
        gamma=settings["gamma"],
        rng=np.random.default_rng(settings["random_streams"]["learner"]),
        actor_learning_rate=settings["actor_learning_rate"],
        critic_learning_rate=settings["critic_learning_rate"],
        batch_size=settings["batch_size"],
        replay_capacity=settings["replay_capacity"],
        tau=settings["tau"],
        policy_delay=settings["policy_delay"],
        target_smoothing_sigma=settings["target_smoothing_sigma_pu"],
        target_smoothing_clip=settings["target_smoothing_clip_pu"],
        layer_sizes=tuple(record["agent"]["layer_sizes"]),
    )
    checkpoints = run_dir / "checkpoints"
    actor = _checked_state(checkpoints / "Actor_td3", agent.actor.state_dict())
    critic = _checked_state(checkpoints / "Critic_td3", agent.critic.state_dict())
    agent.actor.load_state_dict(actor)
    agent.critic.load_state_dict(critic)
    agent.actor.eval()
    agent.critic.eval()
    return agent, task
