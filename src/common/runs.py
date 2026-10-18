"""Create and resolve saved task identity and inference settings.

Current run records separate physical task settings from agent architecture.
Exact historical records remain readable through explicit feeder, observation,
action, reward, and network definitions.
"""

import csv
import json
import math

LEGACY_FIELDS = {"feeder", "dataset", "observation_size", "action_count"}
TASK_FIELDS = {"feeder", "dataset", "algorithm", "observation", "action", "reward", "evaluation"}
ACTION_FIELDS = {"count", "low", "high", "proposal", "execution", "replay", "units"}
HISTORICAL_SHAPES = {
    "ieee13": {"ddpg": (41, 3), "seq_ddpg": (46, 3)},
    "ieee37std": {"ddpg": (114, 2), "seq_ddpg": (118, 2)},
    "ieee37mod": {"ddpg": (117, 5), "seq_ddpg": (124, 5)},
}
HISTORICAL_LAYERS = (400, 300)
HISTORICAL_BOUNDS = (0.90, 1.10)
TD3_SETTING_FIELDS = {
    "seed",
    "random_streams",
    "gamma",
    "actor_learning_rate",
    "critic_learning_rate",
    "batch_size",
    "replay_capacity",
    "replay_sampling",
    "tau",
    "policy_delay",
    "target_smoothing_sigma_normalized",
    "target_smoothing_sigma_pu",
    "target_smoothing_clip_normalized",
    "target_smoothing_clip_pu",
    "exploration_sigma_normalized",
    "exploration_sigma_pu",
    "warmup_interactions",
    "warmup_days",
    "learning_start_interaction",
    "critic_iterations_per_interaction",
    "action_map",
    "initialization",
    "checkpoint_support",
    "training_days",
}


def make_run_record(
    *,
    feeder,
    dataset,
    algorithm,
    observation_name,
    observation_size,
    action_count,
    action_low,
    action_high,
    reward_name,
    reward_parameters,
    layer_sizes,
    benchmark_beta=None,
    agent_settings=None,
):
    """Describe one training task and the network widths needed for inference.

    The action record distinguishes continuous p.u. proposals, executed integer
    positions, and executed p.u. ratios stored in replay. ``benchmark_beta``
    belongs to evaluation and remains separate from reward parameters.
    """
    agent = {"layer_sizes": list(layer_sizes)}
    if agent_settings is not None:
        agent["settings"] = agent_settings
    return {
        "task": {
            "feeder": feeder,
            "dataset": dataset,
            "algorithm": algorithm,
            "observation": {"name": observation_name, "size": observation_size},
            "action": {
                "count": action_count,
                "low": action_low,
                "high": action_high,
                "proposal": "continuous_ratio",
                "execution": "tap_position",
                "replay": "executed_ratio",
                "units": "p.u.",
            },
            "reward": {"name": reward_name, "parameters": reward_parameters},
            "evaluation": (
                {"metric": "voltage_band"}
                if benchmark_beta is None
                else {"metric": "voltage_band", "beta": benchmark_beta}
            ),
        },
        "agent": agent,
    }


def _finite_number(value):
    """Return whether a value is a finite real number but not a Boolean."""
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _positive_integer(value):
    """Return whether a value is a strictly positive built-in integer."""
    return type(value) is int and value > 0


def _validate_td3_settings(settings, action, path):
    """Validate the complete recorded TD3 recipe and its action-unit conversions."""
    if not isinstance(settings, dict) or set(settings) != TD3_SETTING_FIELDS:
        raise ValueError(f"{path}: malformed TD3 agent settings")
    seed = settings["seed"]
    streams = settings["random_streams"]
    action_map = settings["action_map"]
    if (
        type(seed) is not int
        or seed < 0
        or not isinstance(streams, dict)
        or set(streams) != {"day_order", "learner", "torch"}
        or streams["day_order"] != [seed, 1]
        or streams["learner"] != [seed, 2]
        or streams["torch"] != seed
        or not isinstance(action_map, dict)
        or set(action_map) != {"midpoint", "half_range", "normalized_range"}
        or action_map["normalized_range"] != [-1.0, 1.0]
    ):
        raise ValueError(f"{path}: invalid TD3 stream or action-map settings")
    positive_counts = (
        "batch_size",
        "replay_capacity",
        "policy_delay",
        "warmup_interactions",
        "warmup_days",
        "learning_start_interaction",
        "critic_iterations_per_interaction",
        "training_days",
    )
    unit_interval = ("actor_learning_rate", "critic_learning_rate", "tau")
    noises = (
        "target_smoothing_sigma_normalized",
        "target_smoothing_sigma_pu",
        "target_smoothing_clip_normalized",
        "target_smoothing_clip_pu",
        "exploration_sigma_normalized",
        "exploration_sigma_pu",
    )
    if (
        any(not _positive_integer(settings[name]) for name in positive_counts)
        or any(
            not _finite_number(settings[name]) or not 0 < settings[name] <= 1
            for name in unit_interval
        )
        or not _finite_number(settings["gamma"])
        or not 0 <= settings["gamma"] <= 1
        or any(not _finite_number(settings[name]) or settings[name] < 0 for name in noises)
        or settings["replay_sampling"] != "with_replacement"
        or settings["initialization"] != "torch_default_linear"
        or settings["checkpoint_support"] != "inference_only"
    ):
        raise ValueError(f"{path}: invalid TD3 agent settings")
    midpoint = (action["low"] + action["high"]) / 2
    half_range = (action["high"] - action["low"]) / 2
    if (
        not _finite_number(action_map["midpoint"])
        or not _finite_number(action_map["half_range"])
        or not math.isclose(action_map["midpoint"], midpoint, rel_tol=0.0, abs_tol=1e-12)
        or not math.isclose(action_map["half_range"], half_range, rel_tol=0.0, abs_tol=1e-12)
    ):
        raise ValueError(f"{path}: TD3 action map disagrees with action bounds")
    conversions = (
        ("target_smoothing_sigma_normalized", "target_smoothing_sigma_pu"),
        ("target_smoothing_clip_normalized", "target_smoothing_clip_pu"),
        ("exploration_sigma_normalized", "exploration_sigma_pu"),
    )
    if any(
        not math.isclose(
            settings[physical],
            settings[normalized] * half_range,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for normalized, physical in conversions
    ):
        raise ValueError(f"{path}: TD3 noise units disagree with the action map")


def _new_record(record, path):
    """Validate the complete schema and value domains of a current run record."""
    if set(record) != {"task", "agent"}:
        raise ValueError(f"{path}: malformed run record groups")
    task, agent = record["task"], record["agent"]
    if not isinstance(task, dict) or not isinstance(agent, dict):
        raise ValueError(f"{path}: malformed run record groups")
    if set(task) != TASK_FIELDS or set(agent) not in (
        {"layer_sizes"},
        {"layer_sizes", "settings"},
    ):
        raise ValueError(f"{path}: malformed run record fields")
    observation, action, reward = task["observation"], task["action"], task["reward"]
    evaluation = task["evaluation"]
    if (
        not isinstance(observation, dict)
        or set(observation) != {"name", "size"}
        or not isinstance(action, dict)
        or set(action) != ACTION_FIELDS
        or not isinstance(reward, dict)
        or set(reward) != {"name", "parameters"}
        or not isinstance(evaluation, dict)
        or set(evaluation) not in ({"metric"}, {"metric", "beta"})
    ):
        raise ValueError(f"{path}: malformed run record settings")
    widths = agent["layer_sizes"]
    if (
        not isinstance(widths, list)
        or len(widths) != 2
        or any(type(width) is not int or width < 1 for width in widths)
        or type(action["count"]) is not int
        or action["count"] < 1
        or not all(isinstance(action[key], (float, int)) for key in ("low", "high"))
        or not math.isfinite(action["low"])
        or not math.isfinite(action["high"])
        or action["low"] >= action["high"]
        or action["proposal"] != "continuous_ratio"
        or action["execution"] != "tap_position"
        or action["replay"] != "executed_ratio"
        or action["units"] != "p.u."
        or not isinstance(reward["name"], str)
        or not isinstance(reward["parameters"], dict)
        or evaluation["metric"] != "voltage_band"
        or (
            "beta" in evaluation
            and (
                not isinstance(evaluation["beta"], (int, float))
                or not math.isfinite(evaluation["beta"])
                or evaluation["beta"] < 0
            )
        )
    ):
        raise ValueError(f"{path}: invalid action, reward, or agent settings")
    settings = agent.get("settings")
    if settings is not None and not isinstance(settings, dict):
        raise ValueError(f"{path}: malformed agent settings")
    if task["algorithm"] == "seq_td3":
        if settings is None:
            raise ValueError(f"{path}: TD3 agent settings are missing")
        _validate_td3_settings(settings, action, path)
    return task


def write_run_record(directory, record):
    """Validate and write ``run.json`` without replacing an existing record."""
    path = directory / "run.json"
    _new_record(record, path)
    if path.exists():
        raise ValueError(f"{path} already exists")
    directory.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n")


def read_run_record(
    directory, *, feeder, dataset, algorithm, observation_name, observation_size, action_count
):
    """Read a current record or an exact historical identity record.

    The supplied feeder, dataset, algorithm, observation, and action dimensions
    describe the evaluation being requested. They must match saved identity.
    Missing records are accepted only for supported historical IEEE13 DDPG runs.
    """
    path = directory / "run.json"
    if not path.is_file():
        if feeder == "ieee13" and algorithm in {"ddpg", "seq_ddpg"}:
            return None
        raise ValueError(f"{path} is missing for feeder {feeder}")
    try:
        record = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{path}: invalid JSON") from error
    if not isinstance(record, dict):
        raise ValueError(f"{path}: malformed run record")
    if set(record) == LEGACY_FIELDS:
        expected = {
            "feeder": feeder,
            "dataset": dataset,
            "observation_size": observation_size,
            "action_count": action_count,
        }
        if record != expected:
            raise ValueError(f"{path} records {record}, expected {expected}")
        return record
    task = _new_record(record, path)
    expected = {
        "feeder": feeder,
        "dataset": dataset,
        "algorithm": algorithm,
        "observation": {"name": observation_name, "size": observation_size},
        "action_count": action_count,
    }
    actual = {
        "feeder": task["feeder"],
        "dataset": task["dataset"],
        "algorithm": task["algorithm"],
        "observation": task["observation"],
        "action_count": task["action"]["count"],
    }
    if actual != expected:
        raise ValueError(f"{path} records {actual}, expected {expected}")
    return record


def _legacy_beta(path):
    """Recover sequential reward beta from the first historical training row."""
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        first = next(reader, None)
        if reader.fieldnames is None or "beta" not in reader.fieldnames:
            raise ValueError(f"beta column is missing from {path}")
    if first is None:
        raise ValueError(f"training record has no data rows: {path}")
    try:
        beta = float(first["beta"])
    except (TypeError, ValueError) as error:
        raise ValueError(f"beta is invalid in the first data row of {path}") from error
    if not math.isfinite(beta) or beta < 0:
        raise ValueError(f"beta is invalid in the first data row of {path}")
    return beta


def _legacy_settings(run_dir, feeder, algorithm, observation_name, size, count):
    """Reconstruct verified task settings for an exact historical run shape."""
    expected_name = "voltage_observation" if algorithm == "ddpg" else "day_observation"
    expected_shape = HISTORICAL_SHAPES.get(feeder, {}).get(algorithm)
    if (size, count) != expected_shape or observation_name != expected_name:
        raise ValueError(f"unsupported historical task {feeder}/{algorithm}: {size}, {count}")
    if algorithm == "ddpg":
        reward = {
            "name": "voltage_band",
            "parameters": {
                "mad_weight": 0.0,
                "voltage_min": 0.95,
                "voltage_max": 1.05,
                "scale": 1000.0,
            },
        }
        evaluation = {"metric": "voltage_band"}
    else:
        beta = _legacy_beta(run_dir / "train.csv")
        reward = {
            "name": "voltage_band_switching",
            "parameters": {
                "beta": beta,
                "voltage_min": 0.95,
                "voltage_max": 1.05,
                "scale": 1000.0,
            },
        }
        evaluation = {"metric": "voltage_band", "beta": beta}
    return {
        "feeder": feeder,
        "algorithm": algorithm,
        "observation": {"name": expected_name, "size": size},
        "action": {
            "count": count,
            "low": HISTORICAL_BOUNDS[0],
            "high": HISTORICAL_BOUNDS[1],
            "proposal": "continuous_ratio",
            "execution": "tap_position",
            "replay": "executed_ratio",
            "units": "p.u.",
        },
        "reward": reward,
        "evaluation": evaluation,
    }


def resolve_run_settings(
    run_dir,
    *,
    feeder,
    dataset,
    algorithm,
    observation_name,
    observation_size,
    action_count,
):
    """Return saved task settings and network widths without loading checkpoints.

    Current records supply their own settings. Historical records use only the
    explicitly supported shapes, layer widths, action bounds, reward definitions,
    and, for sequential runs, beta recovered from ``train.csv``.
    """
    record = read_run_record(
        run_dir,
        feeder=feeder,
        dataset=dataset,
        algorithm=algorithm,
        observation_name=observation_name,
        observation_size=observation_size,
        action_count=action_count,
    )
    if record is None or set(record) == LEGACY_FIELDS:
        task = _legacy_settings(
            run_dir, feeder, algorithm, observation_name, observation_size, action_count
        )
        return task, HISTORICAL_LAYERS
    return record["task"], tuple(record["agent"]["layer_sizes"])
