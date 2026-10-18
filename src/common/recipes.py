"""Prepare aligned saved recipe evidence for sequential learner reports."""

import json


def _read_object(path):
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read recipe evidence from {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"recipe evidence is not an object: {path}")
    return value


def _entry(value, provenance):
    return {"value": value, "provenance": provenance}


def _join_widths(widths):
    return "/".join(str(width) for width in widths)


def _count(value, unit):
    suffix = unit if value == 1 else f"{unit}s"
    return f"{value} {suffix}"


def _saved_ddpg(settings, name, transform=lambda value: value):
    entry = settings[name]
    return _entry(transform(entry["value"]), entry["provenance"])


def _stream_description(seed, record):
    streams = record["agent"]["settings"]["random_streams"]
    return (
        f"seed {seed}: day_order {streams['day_order']}; "
        f"learner {streams['learner']}; torch {streams['torch']}"
    )


def _td3_evidence(results, seeds):
    if len(results.run_names) != len(seeds) or len(results.learned_labels) != len(seeds):
        raise ValueError("TD3 selected seeds do not align with result runs")
    records = []
    counts = []
    completed = []
    for seed, run, label in zip(seeds, results.run_names, results.learned_labels, strict=True):
        directory = results.run_directories[run]
        record = _read_object(directory / "run.json")
        update_counts = _read_object(directory / "counts.json")
        settings = record["agent"]["settings"]
        expected_streams = {
            "day_order": [seed, 1],
            "learner": [seed, 2],
            "torch": seed,
        }
        if settings["seed"] != seed or settings["random_streams"] != expected_streams:
            raise ValueError(f"TD3 seed or random streams disagree for {run}")
        if results.update_counts[label] != update_counts:
            raise ValueError(f"TD3 update counts disagree for {run}")
        completed_days = results.training_counts[label]
        if completed_days != update_counts["days"]:
            raise ValueError(f"TD3 completed days disagree for {run}")
        records.append(record)
        counts.append(update_counts)
        completed.append(completed_days)

    first = records[0]
    first_settings = first["agent"]["settings"]
    shared_names = tuple(name for name in first_settings if name not in {"seed", "random_streams"})
    for run, record in zip(results.run_names[1:], records[1:], strict=True):
        current = record["agent"]["settings"]
        if set(current) != set(first_settings) or any(
            current[name] != first_settings[name] for name in shared_names
        ):
            raise ValueError(f"TD3 shared recipe settings disagree for {run}")
        if record["agent"]["layer_sizes"] != first["agent"]["layer_sizes"]:
            raise ValueError(f"TD3 layer sizes disagree for {run}")
        if record["task"] != first["task"]:
            raise ValueError(f"TD3 task settings disagree for {run}")
    if len(set(completed)) != 1:
        raise ValueError("TD3 completed days disagree across selected runs")
    return first, first_settings, records, counts, completed[0]


def _ddpg_evidence(results, seeds, dataset):
    if len(results.run_names) != len(seeds) or len(results.learned_labels) != len(seeds):
        raise ValueError("DDPG selected seeds do not align with result runs")
    path = results.root / "results" / dataset / "seq_ddpg" / "recipe.json"
    recipe = _read_object(path)
    if recipe.get("dataset") != dataset:
        raise ValueError("saved DDPG recipe describes a different dataset")
    inventory = recipe.get("run_names")
    if not isinstance(inventory, list):
        raise ValueError("saved DDPG recipe has no run inventory")
    for run in results.run_names:
        if run not in inventory:
            raise ValueError(f"{run}: saved DDPG recipe has no description for it")
    tasks = [results.run_tasks[label] for label in results.learned_labels]
    if any(task != tasks[0] for task in tasks[1:]):
        raise ValueError("DDPG task settings disagree across selected runs")
    return recipe["settings"], tasks[0]


def prepare_recipe_comparison(td3_results, ddpg_results, selected_seeds):
    """Return aligned TD3 and DDPG recipe rows from saved experiment evidence."""
    seeds = tuple(selected_seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("selected seeds must be distinct and non-empty")
    if tuple(td3_results.settings["seeds"]) != seeds:
        raise ValueError("TD3 results do not match the selected seed order")
    if tuple(ddpg_results.settings["seeds"]) != seeds:
        raise ValueError("DDPG results do not match the selected seed order")
    dataset = td3_results.settings["dataset"]
    if ddpg_results.settings["dataset"] != dataset:
        raise ValueError("TD3 and DDPG datasets differ")

    record, settings, records, counts, completed_days = _td3_evidence(td3_results, seeds)
    ddpg, ddpg_task = _ddpg_evidence(ddpg_results, seeds, dataset)
    task = record["task"]
    widths = record["agent"]["layer_sizes"]
    ddpg_widths = ddpg["layer_sizes"]["value"]

    td3_recipe = {
        "hidden_layers": _entry(
            f"actor {_join_widths(widths)}; two critic branches {_join_widths(widths)}",
            "reconstructed",
        ),
        "warmup": _entry(
            f"{_count(settings['warmup_days'], 'day')}; "
            f"{_count(settings['warmup_interactions'], 'interaction')}",
            "recorded",
        ),
        "learning_start_interaction": _entry(settings["learning_start_interaction"], "recorded"),
        "layer_normalization": _entry("none", "reconstructed"),
        "gradient_clipping": _entry("none", "reconstructed"),
        "initialization": _entry(settings["initialization"], "recorded"),
        "actor_learning_rate": _entry(settings["actor_learning_rate"], "recorded"),
        "critic_learning_rate": _entry(settings["critic_learning_rate"], "recorded"),
        "critic_count": _entry(2, "reconstructed"),
        "batch_size": _entry(settings["batch_size"], "recorded"),
        "replay_capacity": _entry(settings["replay_capacity"], "recorded"),
        "replay_sampling": _entry(settings["replay_sampling"], "recorded"),
        "tau": _entry(settings["tau"], "recorded"),
        "gamma": _entry(settings["gamma"], "recorded"),
        "policy_delay": _entry(settings["policy_delay"], "recorded"),
        "exploration_noise": _entry(
            f"Gaussian sigma {settings['exploration_sigma_normalized']} normalized "
            f"({settings['exploration_sigma_pu']} p.u.) after warmup",
            "recorded",
        ),
        "uniform_action_share": _entry("warmup 1.0; after warmup 0.0", "reconstructed"),
        "target_smoothing": _entry(
            f"Gaussian sigma {settings['target_smoothing_sigma_normalized']} normalized "
            f"({settings['target_smoothing_sigma_pu']} p.u.); clip "
            f"{settings['target_smoothing_clip_normalized']} normalized "
            f"({settings['target_smoothing_clip_pu']} p.u.)",
            "recorded",
        ),
        "critic_weight_decay": _entry("none", "reconstructed"),
        "checkpoint_cadence_days": _entry(200, "reconstructed"),
        "reward_name": _entry(task["reward"]["name"], "recorded"),
        "reward_parameters": _entry(task["reward"]["parameters"], "recorded"),
        "benchmark_beta": _entry(task["evaluation"]["beta"], "recorded"),
        "completed_days": _entry(completed_days, "recorded"),
        "observation_size": _entry(task["observation"]["size"], "recorded"),
        "action_bounds_pu": _entry([task["action"]["low"], task["action"]["high"]], "recorded"),
        "seeds_and_random_streams": _entry(
            [_stream_description(seed, saved) for seed, saved in zip(seeds, records, strict=True)],
            "recorded",
        ),
        "update_counts_by_seed": _entry(
            [f"seed {seed}: {value}" for seed, value in zip(seeds, counts, strict=True)],
            "recorded",
        ),
    }

    def ddpg_noise(value):
        return (
            f"{value['process']}; sigma {value['start']} to floor {value['floor']}; "
            f"decay {value['decay']}"
        )

    ddpg_recipe = {
        "hidden_layers": _entry(
            f"actor {_join_widths(ddpg_widths)}; critic {_join_widths(ddpg_widths)} "
            f"with a merged {ddpg_widths[1]}-unit layer",
            "reconstructed",
        ),
        "warmup": _saved_ddpg(
            ddpg,
            "warmup",
            lambda value: (
                f"{_count(value['days'], 'day')}; {_count(value['interactions'], 'interaction')}"
            ),
        ),
        "learning_start_interaction": _saved_ddpg(ddpg, "learning_start_interaction"),
        "layer_normalization": _saved_ddpg(ddpg, "layer_normalization"),
        "gradient_clipping": _saved_ddpg(
            ddpg,
            "gradient_clipping",
            lambda value: (
                f"actor max norm {value['actor_max_norm']}; "
                f"critic max norm {value['critic_max_norm']}"
            ),
        ),
        "initialization": _saved_ddpg(ddpg, "initialization"),
        "actor_learning_rate": _saved_ddpg(ddpg, "actor_learning_rate"),
        "critic_learning_rate": _saved_ddpg(ddpg, "critic_learning_rate"),
        "critic_count": _saved_ddpg(ddpg, "critic_count"),
        "batch_size": _saved_ddpg(ddpg, "batch_size"),
        "replay_capacity": _saved_ddpg(ddpg, "replay_capacity"),
        "replay_sampling": _saved_ddpg(ddpg, "replay_sampling"),
        "tau": _saved_ddpg(ddpg, "tau"),
        "gamma": _saved_ddpg(ddpg, "gamma"),
        "policy_delay": _entry(1, "reconstructed"),
        "exploration_noise": _saved_ddpg(ddpg, "noise_schedule_normalized", ddpg_noise),
        "uniform_action_share": _saved_ddpg(
            ddpg,
            "uniform_action_share",
            lambda value: f"warmup {value['warmup']}; after warmup {value['after_warmup']}",
        ),
        "target_smoothing": _entry("none", "reconstructed"),
        "critic_weight_decay": _saved_ddpg(ddpg, "critic_weight_decay"),
        "checkpoint_cadence_days": _saved_ddpg(ddpg, "checkpoint_cadence_days"),
        "reward_name": _entry(ddpg_task["reward"]["name"], "recorded"),
        "reward_parameters": _saved_ddpg(ddpg, "reward_parameters"),
        "benchmark_beta": _saved_ddpg(ddpg, "benchmark_beta"),
        "completed_days": _saved_ddpg(ddpg, "completed_days"),
        "observation_size": _saved_ddpg(ddpg, "observation_size"),
        "action_bounds_pu": _saved_ddpg(ddpg, "action_bounds_pu"),
        "seeds_and_random_streams": _entry(
            [f"seed {seed}: random streams not recorded" for seed in seeds],
            "reconstructed",
        ),
    }
    return {"TD3": td3_recipe, "DDPG": ddpg_recipe}
