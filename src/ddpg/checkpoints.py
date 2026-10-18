"""Reconstruct DDPG inference from saved run settings and checkpoints.

Evaluation validates experiment identity, action semantics, network widths,
checkpoint keys, and tensor shapes before it writes any result output.
"""

import torch

from common.runs import resolve_run_settings
from ddpg.ddpg_vvo import Agent


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
    """Return an inference agent reconstructed from one compatible saved run.

    Current records provide task settings and hidden widths. Supported historical
    records recover verified widths and reward definitions through
    ``resolve_run_settings``. Both actor and critic checkpoints must match the
    reconstructed tensors, although controller inference uses the actor.
    """
    task, widths = resolve_run_settings(
        run_dir,
        feeder=feeder,
        dataset=dataset,
        algorithm=algorithm,
        observation_name=observation_name,
        observation_size=observation_size,
        action_count=action_count,
    )
    action = task["action"]
    if (action["low"], action["high"]) != tuple(action_bounds):
        raise ValueError(f"run action bounds disagree with supported feeder bounds: {run_dir}")
    if algorithm == "seq_ddpg":
        beta = task["evaluation"].get("beta")
        if beta is None:
            raise ValueError(f"benchmark beta is missing from run record: {run_dir}")
    agent = Agent(
        action_low=action["low"],
        action_high=action["high"],
        alpha=1e-4,
        beta=1e-3,
        input_dims=observation_size,
        tau=1e-3,
        n_actions=action_count,
        layer1_size=widths[0],
        layer2_size=widths[1],
    )
    checkpoints = run_dir / "checkpoints"
    actor = _checked_state(checkpoints / "Actor_ddpg", agent.actor.state_dict())
    critic = _checked_state(checkpoints / "Critic_ddpg", agent.critic.state_dict())
    agent.actor.load_state_dict(actor)
    agent.critic.load_state_dict(critic)
    agent.actor.eval()
    agent.critic.eval()
    return agent, task
