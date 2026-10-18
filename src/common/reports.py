"""Display notebook reports from validated, aligned experiment results.

These functions read the supplied ``ExperimentResults`` and feeder configuration.
They do not train, evaluate, change saved outputs, or claim performance beyond
the selected split.
"""

import json
import statistics

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Markdown, display

from common.display import show_table
from common.outputs import daily_summary, paired_counts, static_summary
from common.plots import (
    hourly_detail_rows,
    plot_cost_feasibility,
    plot_day_ratios,
    plot_static_taps,
    tap_ratio_fields,
)
from feeder.config import read_config
from feeder.taps import N_TAPS

FIXED_COLORS = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c"}
REFERENCE_COLORS = {"Neutral": "#7f7f7f", "RegControl": "#d62728"}


def _colors(results):
    """Assign stable colors to default seeds and both reference controllers."""
    fallback = plt.get_cmap("tab10")
    colors = {
        label: FIXED_COLORS.get(seed, fallback(index % 10))
        for index, (seed, label) in enumerate(
            zip(results.settings["seeds"], results.learned_labels, strict=True)
        )
    }
    colors.update(REFERENCE_COLORS)
    return colors


def _record_rows(results):
    """Build rows from saved reward, benchmark, architecture, and history data."""
    unit = "days" if results.settings["protocol"] == "sequential" else "episodes"
    rows = []
    for label in results.learned_labels:
        task = results.run_tasks[label]
        rows.append(
            {
                "Run": label,
                "Reward": task["reward"]["name"],
                "Reward parameters": task["reward"]["parameters"],
                "Benchmark beta": task["evaluation"].get("beta"),
                "Layer sizes": results.layer_sizes[label],
                f"Completed training {unit}": (
                    results.training_counts[label]
                    if results.training_counts[label] is not None
                    else "unavailable"
                ),
                "Update counts": results.update_counts[label] or "unavailable",
            }
        )
    return rows


def show_training_summary(results):
    """Display recorded run settings, completed counts, and reward curves.

    Sequential episode counts represent complete days and are also reported as
    24 interactions per completed day. Curves include exploratory training data;
    they are not held-out performance or evidence of convergence.
    """
    sequential = results.settings["protocol"] == "sequential"
    title = (
        "Recorded reward and benchmark beta by run"
        if sequential
        else "Recorded training settings by run"
    )
    show_table(_record_rows(results), title=title)
    window_limit = 50 if sequential else 200
    figure, axis = plt.subplots(figsize=(8, 4), dpi=100)
    plotted = False
    completed = {}
    colors = _colors(results)
    for name, label in zip(results.run_names, results.learned_labels, strict=True):
        rows = results.training_records[label]
        completed[name] = results.training_counts[label]
        if not rows:
            print(f"{name}: training curve unavailable")
            continue
        episodes = np.asarray([int(row["episode"]) for row in rows])
        rewards = np.asarray([float(row["score"]) for row in rows])
        window = min(window_limit, len(rows))
        moving = np.convolve(rewards, np.ones(window) / window, mode="valid")
        axis.plot(episodes, rewards, color=colors[label], alpha=0.14, linewidth=0.5)
        axis.plot(
            episodes[window - 1 :],
            moving,
            color=colors[label],
            label=f"{label} moving mean (window {window})",
        )
        if sequential:
            print(
                f"{name}: {len(rows)} completed episodes, {24 * len(rows)} environment interactions"
            )
        plotted = True
    if plotted:
        axis.set_title("Sequential training reward" if sequential else "One-shot training reward")
        axis.set_xlabel(
            "Episode / complete day (count)"
            if sequential
            else "Episode / environment interaction (count)"
        )
        axis.set_ylabel("Training reward (dimensionless)")
        axis.legend()
        axis.grid(alpha=0.25)
        figure.tight_layout()
        plt.show()
    else:
        plt.close(figure)
    if not sequential:
        print("Completed training episodes:", completed)


def _summaries(results):
    """Calculate one aggregate physical summary per aligned controller result."""
    if results.settings["protocol"] == "one_shot":
        return {
            label: static_summary(results.result_rows[label]) for label in results.method_labels
        }
    beta = results.settings["beta"]
    return {
        label: daily_summary(results.daily_rows[label], beta) for label in results.method_labels
    }


def _seed_statistics(results, summaries, columns, formats):
    """Display mean and sample standard deviation across trained seed runs.

    These values describe variation across run-level aggregates. They are not
    across-day variation and do not form confidence intervals.
    """
    if len(results.learned_labels) < 2:
        print("Across-seed summary skipped because one seed is selected.")
        return
    rows = []
    for name, function in (
        ("Across-seed mean", statistics.fmean),
        ("Sample standard deviation", statistics.stdev),
    ):
        row = {f"{results.algorithm_label} statistic": name}
        for column, format_spec in zip(columns, formats, strict=True):
            value = function(summaries[label][column] for label in results.learned_labels)
            row[column] = format(value, format_spec)
        rows.append(row)
    show_table(rows, title=f"Across-seed {results.algorithm_label} summary")


def _controller_explanation(results, *, sequential):
    """Display feeder-specific measurement and operational-cost definitions."""
    count = len(results.simulator.measurement_names)
    convention = read_config(results.settings["feeder"]).voltage_convention
    references = (
        "Neutral holds every regulator ratio at 1.0. RegControl uses the feeder's native "
        "regulator control."
    )
    if sequential:
        definition = (
            "Hourly voltage cost is the mean squared excursion outside [0.95, 1.05] p.u. "
            f"over {count} monitored {convention} voltage measurements; in-band measurements "
            "contribute zero. Daily voltage cost sums 24 hourly costs, and the table averages "
            "these daily totals across evaluation days. Lower cost is better. Beta weights tap "
            "movement; it is not a monetary price."
        )
    else:
        definition = (
            "Hourly voltage cost is the mean squared excursion outside [0.95, 1.05] p.u. "
            f"over {count} monitored {convention} voltage measurements; in-band measurements "
            "contribute zero. Lower cost is better."
        )
    display(Markdown(references + "\n\n" + definition))


def _show_controller_comparison_oneshot(results):
    """Display static aggregates, seed statistics, and cost-feasibility plots."""
    summaries = _summaries(results)
    _controller_explanation(results, sequential=False)
    rows = [
        {
            "Method": label,
            "Mean hourly voltage cost": f"{values['Mean hourly voltage cost']:.4g}",
            "Feasible scenarios (%)": f"{values['Feasible scenarios (%)']:.1f}",
            "Mean violating voltage measurements/scenario": (
                f"{values['Mean violating voltage measurements/scenario']:.2f}"
            ),
            "Scenarios": values["Scenarios"],
        }
        for label, values in summaries.items()
    ]
    show_table(rows, title=f"Overall results on {results.settings['split']}")
    columns = (
        "Mean hourly voltage cost",
        "Feasible scenarios (%)",
        "Mean violating voltage measurements/scenario",
    )
    _seed_statistics(results, summaries, columns, (".4g", ".1f", ".2f"))
    _figure, axes = plot_cost_feasibility(
        summaries,
        results.method_labels,
        results.settings["split"],
        _colors(results),
    )
    scale = axes[0].get_yscale()
    plt.show()
    if scale == "log":
        display(
            Markdown(
                "The voltage-cost panel uses a log scale because every displayed cost "
                "is positive. Log spacing shows multiplicative differences."
            )
        )
    else:
        display(
            Markdown(
                "The voltage-cost panel uses a linear scale because at least one "
                "displayed cost is zero."
            )
        )


def _show_controller_comparison_seq(results):
    """Display daily aggregates and seed statistics under the selected beta."""
    summaries = _summaries(results)
    _controller_explanation(results, sequential=True)
    rows = [
        {
            "Method": label,
            "Mean daily combined cost": f"{values['Mean daily combined cost']:.6g}",
            "Mean daily voltage cost": f"{values['Mean daily voltage cost']:.6g}",
            "Mean daily switching cost": f"{values['Mean daily switching cost']:.6g}",
            "Mean tap steps/day": f"{values['Mean tap steps/day']:.3f}",
            "Feasible hours (%)": f"{values['Feasible hours (%)']:.1f}",
            "Days": values["Days"],
        }
        for label, values in summaries.items()
    ]
    show_table(rows, title=f"Overall results on {results.settings['split']}")
    columns = (
        "Mean daily combined cost",
        "Mean daily voltage cost",
        "Mean daily switching cost",
        "Mean tap steps/day",
        "Feasible hours (%)",
    )
    _seed_statistics(results, summaries, columns, (".6g", ".6g", ".6g", ".3f", ".1f"))


def show_controller_comparison(results):
    """Display physical summaries and across-seed variation for all methods.

    One-shot summaries average hourly scenarios. Sequential summaries first sum
    each complete day, then average those daily totals across evaluated days.
    """
    if results.settings["protocol"] == "one_shot":
        return _show_controller_comparison_oneshot(results)
    return _show_controller_comparison_seq(results)


def show_paired_counts(results):
    """Display exact paired lower, equal, and higher counts against references.

    Static rows pair matching scenarios; sequential rows pair matching days.
    Equality is exact and the counts do not express practical equivalence or
    effect size.
    """
    sequential = results.settings["protocol"] == "sequential"
    source = results.daily_rows if sequential else results.result_rows
    column = "daily_cost" if sequential else "cost"
    rows = []
    for label in results.learned_labels:
        for reference in ("Neutral", "RegControl"):
            lower, equal, higher = paired_counts(source[label], source[reference], column)
            suffix = "days" if sequential else "scenarios"
            rows.append(
                {
                    f"{results.algorithm_label} run": label,
                    "Reference": reference,
                    f"Lower-cost {suffix}": lower,
                    f"Equal-cost {suffix}": equal,
                    f"Higher-cost {suffix}": higher,
                }
            )
    title = (
        "Paired combined-cost counts on matching days"
        if sequential
        else "Paired voltage-cost counts on matching scenarios"
    )
    show_table(rows, title=title)


def _comparison_metrics(results, label):
    """Return per-run daily metrics and an explicit feasible-hour count."""
    rows = results.daily_rows[label]
    summary = daily_summary(rows, results.settings["beta"])
    feasible = sum(int(row["feasible_hours"]) for row in rows)
    return {
        "Mean daily voltage cost": summary["Mean daily voltage cost"],
        "Mean tap steps/day": summary["Mean tap steps/day"],
        "Mean daily switching cost": summary["Mean daily switching cost"],
        "Mean daily combined cost": summary["Mean daily combined cost"],
        "Feasible hours": feasible,
    }


def show_algorithm_comparison(left, right):
    """Compare two sequential learners on matching saved days and seeds.

    The comparison uses common physical costs only. Paired differences match
    equal seed numbers and day identities. Matching exogenous loads and neutral
    initial taps do not make the learners' observation trajectories identical.
    """
    if left.settings["protocol"] != "sequential" or right.settings["protocol"] != "sequential":
        raise ValueError("algorithm comparison requires sequential results")
    checks = (
        (left.settings["dataset"], right.settings["dataset"], "dataset"),
        (left.settings["split"], right.settings["split"], "split"),
        (left.settings["beta"], right.settings["beta"], "beta"),
        (left.settings["seeds"], right.settings["seeds"], "seeds"),
        (
            tuple(day.id for day in left.evaluation_days),
            tuple(day.id for day in right.evaluation_days),
            "days",
        ),
    )
    for left_value, right_value, name in checks:
        if left_value != right_value:
            raise ValueError(f"algorithm comparison requires matching {name}")
    if left.algorithm_label == right.algorithm_label:
        raise ValueError("algorithm comparison requires different algorithm labels")

    metric_names = (
        "Mean daily voltage cost",
        "Mean tap steps/day",
        "Mean daily switching cost",
        "Mean daily combined cost",
    )
    denominator = len(left.evaluation_days) * 24
    per_algorithm = {}
    rows = []
    paired_rows = []
    for results in (left, right):
        per_algorithm[results.algorithm_label] = []
        for seed, label in zip(results.settings["seeds"], results.learned_labels, strict=True):
            values = _comparison_metrics(results, label)
            per_algorithm[results.algorithm_label].append(values)
            rows.append(
                {
                    "Algorithm": results.algorithm_label,
                    "Seed": seed,
                    **{name: f"{values[name]:.6g}" for name in metric_names},
                    "Feasible hours": f"{values['Feasible hours']} / {denominator}",
                }
            )
    show_table(rows, title="Per-seed sequential algorithm comparison")

    aggregate_rows = []
    for algorithm, values in per_algorithm.items():
        row = {"Algorithm": algorithm}
        for name in (*metric_names, "Feasible hours"):
            selected = [entry[name] for entry in values]
            mean = statistics.fmean(selected)
            spread = statistics.stdev(selected) if len(selected) > 1 else None
            text = f"{mean:.6g} ± {spread:.6g}" if spread is not None else f"{mean:.6g} ± n/a"
            row[name] = f"{text} / {denominator}" if name == "Feasible hours" else text
        aggregate_rows.append(row)
    show_table(aggregate_rows, title="Across-seed mean ± sample standard deviation")

    for seed, left_label, right_label in zip(
        left.settings["seeds"], left.learned_labels, right.learned_labels, strict=True
    ):
        left_rows = left.daily_rows[left_label]
        right_rows = right.daily_rows[right_label]
        left_costs = [float(row["daily_cost"]) for row in left_rows]
        right_costs = [float(row["daily_cost"]) for row in right_rows]
        lower = sum(a < b for a, b in zip(left_costs, right_costs, strict=True))
        equal = sum(a == b for a, b in zip(left_costs, right_costs, strict=True))
        higher = sum(a > b for a, b in zip(left_costs, right_costs, strict=True))
        paired_rows.append(
            {
                "Seed": seed,
                f"{left.algorithm_label} lower": lower,
                "Equal": equal,
                f"{left.algorithm_label} higher": higher,
                f"Mean {left.algorithm_label} − {right.algorithm_label}": (
                    f"{statistics.fmean(a - b for a, b in zip(left_costs, right_costs)):.6g}"
                ),
            }
        )
    show_table(paired_rows, title="Paired daily combined-cost counts")
    display(
        Markdown(
            "Pairs share exogenous loads and neutral initial taps. Their observation "
            "trajectories can differ after controllers execute different taps."
        )
    )


def show_recipe_comparison(recipes):
    """Display ordered saved recipe values with recorded provenance."""
    if not isinstance(recipes, dict) or not recipes:
        raise ValueError("recipes must be a non-empty ordered mapping")
    setting_names = []
    for recipe in recipes.values():
        if not isinstance(recipe, dict):
            raise ValueError("each recipe must be a settings mapping")
        for name in recipe:
            if name not in setting_names:
                setting_names.append(name)
    rows = []
    for name in setting_names:
        row = {"Setting": name}
        for label, recipe in recipes.items():
            if name not in recipe:
                row[label] = "not recorded"
                continue
            entry = recipe[name]
            if (
                not isinstance(entry, dict)
                or set(entry) != {"value", "provenance"}
                or entry["provenance"] not in {"recorded", "reconstructed"}
            ):
                raise ValueError(f"invalid recipe entry for {label}/{name}")
            value = entry["value"]
            rendered = (
                json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else str(value)
            )
            row[label] = f"{rendered} ({entry['provenance']})"
        rows.append(row)
    show_table(rows, title="Recorded and reconstructed learner recipes")


def show_tap_sample(results):
    """Display five static loading ranks and the complete executed-ratio figure.

    Sorted multipliers order independent scenarios for presentation. Connecting
    lines do not represent a trajectory or continuous executed action.
    """
    if results.settings["protocol"] != "one_shot":
        raise ValueError("show_tap_sample requires one-shot results")
    ordered = sorted(
        enumerate(results.evaluation_scenarios),
        key=lambda item: (float(item[1].factors[0]), item[0]),
    )
    ranked = [ordered[int(value * (len(ordered) - 1))][1] for value in (0, 0.25, 0.5, 0.75, 1)]
    sampled = [
        next(
            scenario
            for scenario in results.evaluation_scenarios
            if float(scenario.factors[0]) == float(rank.factors[0])
        )
        for rank in ranked
    ]
    lookups = {
        label: {row["scenario"]: row for row in rows} for label, rows in results.result_rows.items()
    }
    rows = []
    for scenario in sampled:
        for label in results.method_labels:
            source = lookups[label][scenario.id]
            indices = [int(source[f"k{number}"]) for number in range(1, results.action_count + 1)]
            rows.append(
                {
                    "Scenario": scenario.id,
                    "Loading multiplier": float(scenario.factors[0]),
                    "Method": label,
                    **tap_ratio_fields(indices),
                }
            )
    show_table(rows, title="Executed ratios at five loading-multiplier ranks", max_height=520)
    labels = (*results.learned_labels, "RegControl")
    plot_static_taps(
        results.evaluation_scenarios,
        results.result_rows,
        labels,
        _colors(results),
        results.action_count,
    )
    plt.show()
    display(
        Markdown(
            "The x-axis orders independent scenarios by loading multiplier, not time. "
            "Connecting lines aid reading; they are not trajectories or continuous actions."
        )
    )


def show_day(results, day, hourly_details):
    """Display one selected sequential day without changing saved results.

    Figures use the saved hourly rows for every method. Changing the selected day
    or detail-table switch changes presentation only and launches no evaluation.
    """
    if results.settings["protocol"] != "sequential":
        raise ValueError("show_day requires sequential results")
    if (
        isinstance(day, bool)
        or not isinstance(day, int)
        or not 0 <= day < len(results.evaluation_days)
    ):
        raise ValueError(f"day must be an integer from 0 to {len(results.evaluation_days) - 1}")
    if not isinstance(hourly_details, bool):
        raise ValueError("hourly_details must be a Boolean value")
    selected_day = results.evaluation_days[day]
    selected_day_id = selected_day.id
    print(f"Selected day: {selected_day_id}")
    hours = np.arange(24)
    multipliers = np.asarray([float(scenario.factors[0]) for scenario in selected_day.hours])
    selected = {
        label: [row for row in results.hourly_rows[label] if row["day"] == selected_day_id]
        for label in results.method_labels
    }
    colors = _colors(results)
    plot_day_ratios(
        selected_day_id,
        multipliers,
        selected,
        results.method_labels,
        colors,
        results.action_count,
    )
    plt.show()

    figure, axis = plt.subplots(figsize=(10, 3.4), dpi=100)
    for label in results.method_labels:
        steps = [int(row["tap_steps"]) for row in selected[label]]
        axis.step(
            hours,
            steps,
            where="mid",
            marker="o",
            markersize=2.5,
            color=colors[label],
            label=label,
        )
    axis.set_title(f"Hourly tap movement on {selected_day_id}")
    axis.set_xlabel("Hour (count)")
    axis.set_ylabel("Tap steps in the hour (count)")
    axis.set_xticks(hours)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7, ncol=3)
    figure.tight_layout()
    plt.show()
    display(Markdown(f"Switching cost is tap steps times beta ({results.settings['beta_text']})."))

    focused = (*results.learned_labels, "RegControl")
    figure, axes = plt.subplots(2, 1, figsize=(10, 6.2), dpi=100, sharex=True)
    for label in results.method_labels:
        voltage = [float(row["voltage_cost"]) for row in selected[label]]
        axes[0].step(
            hours,
            voltage,
            where="mid",
            marker="o",
            markersize=2.5,
            color=colors[label],
            label=label,
        )
    for label in focused:
        voltage = [float(row["voltage_cost"]) for row in selected[label]]
        axes[1].step(
            hours,
            voltage,
            where="mid",
            marker="o",
            markersize=2.5,
            color=colors[label],
            label=label,
        )
    axes[0].set_title(f"Hourly voltage cost on {selected_day_id}: all methods")
    axes[1].set_title(f"Focused view: {results.algorithm_label} and RegControl")
    axes[0].set_ylabel("Voltage cost")
    axes[1].set_ylabel("Voltage cost")
    axes[1].set_xlabel("Hour (count)")
    axes[1].set_xticks(hours)
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7, ncol=3)
    figure.tight_layout()
    plt.show()
    display(
        Markdown(
            "The upper panel keeps every method. The lower panel repeats the controller "
            "series without Neutral so their smaller differences remain readable."
        )
    )
    details = hourly_detail_rows(selected, results.method_labels, results.action_count)
    if hourly_details:
        display(
            Markdown(
                "Tap indices run from 0 to 32. Convert an index with "
                "`ratio = 0.90 + 0.00625 × index`."
            )
        )
        show_table(
            details,
            title=f"Hourly tap settings and costs on {selected_day_id}",
            max_height=520,
        )
    else:
        print("Hourly detail table hidden. Set SHOW_HOURLY_DETAILS = True to display it.")


def _relation(left, right):
    if left < right:
        return "lower than"
    if left > right:
        return "higher than"
    return "equal to"


def _budget_text(results):
    """Describe recorded completed training episodes or their absence by run."""
    values = [results.training_counts[label] for label in results.learned_labels]
    sequential = results.settings["protocol"] == "sequential"
    if all(value is not None for value in values) and len(set(values)) == 1:
        unit = "complete training days" if sequential else "training episodes"
        return f"{values[0]} {unit} per seed"
    unit = " days" if sequential else ""
    return ", ".join(
        f"{label}: {value if value is not None else 'unavailable'}{unit}"
        for label, value in zip(results.learned_labels, values, strict=True)
    )


def _show_conclusions_oneshot(results):
    """Describe one-shot learned aggregates against both static references."""
    summaries = _summaries(results)
    split = results.settings["split"]
    budget = _budget_text(results)
    costs = [summaries[label]["Mean hourly voltage cost"] for label in results.learned_labels]
    feasible = [summaries[label]["Feasible scenarios (%)"] for label in results.learned_labels]
    mean_cost = statistics.fmean(costs)
    mean_feasible = statistics.fmean(feasible)
    if len(results.learned_labels) >= 2:
        seed_result = (
            f"Across {len(results.learned_labels)} independent seeds, "
            f"{results.algorithm_label} mean hourly "
            f"voltage cost is {mean_cost:.4g} with sample standard deviation "
            f"{statistics.stdev(costs):.4g}."
        )
        feasibility = (
            f"Across {len(results.learned_labels)} independent seeds, "
            f"{results.algorithm_label} feasible "
            f"scenarios average {mean_feasible:.1f}% with sample standard deviation "
            f"{statistics.stdev(feasible):.1f} percentage points"
        )
    else:
        seed_result = (
            f"The selected {results.algorithm_label} run has mean hourly voltage cost "
            f"{mean_cost:.4g}."
        )
        feasibility = (
            f"The selected {results.algorithm_label} run has "
            f"{mean_feasible:.1f}% feasible scenarios"
        )
    neutral = summaries["Neutral"]
    regcontrol = summaries["RegControl"]
    lines = [
        f"The {split} evaluation covers {len(results.evaluation_scenarios)} independent "
        f"scenarios for {len(results.learned_labels)} {results.algorithm_label} seed"
        f"{'s' if len(results.learned_labels) != 1 else ''} and {budget}.",
        seed_result
        + " This is "
        + f"{_relation(mean_cost, neutral['Mean hourly voltage cost'])} Neutral "
        + f"({neutral['Mean hourly voltage cost']:.4g}) and "
        + f"{_relation(mean_cost, regcontrol['Mean hourly voltage cost'])} RegControl "
        + f"({regcontrol['Mean hourly voltage cost']:.4g}).",
        feasibility
        + "; this is "
        + f"{_relation(mean_feasible, neutral['Feasible scenarios (%)'])} Neutral "
        + f"({neutral['Feasible scenarios (%)']:.1f}%) and "
        + f"{_relation(mean_feasible, regcontrol['Feasible scenarios (%)'])} RegControl "
        + f"({regcontrol['Feasible scenarios (%)']:.1f}%).",
        f"These results are limited to the {split} split, this synthetic process, "
        "and the recorded training budget.",
    ]
    display(Markdown("\n\n".join(lines)))


def _show_conclusions_seq(results):
    """Describe sequential learned cost, movement, and feasibility aggregates."""
    summaries = _summaries(results)
    split = results.settings["split"]
    budget = _budget_text(results)
    combined = [summaries[label]["Mean daily combined cost"] for label in results.learned_labels]
    voltage = [summaries[label]["Mean daily voltage cost"] for label in results.learned_labels]
    steps = [summaries[label]["Mean tap steps/day"] for label in results.learned_labels]
    feasible = [summaries[label]["Feasible hours (%)"] for label in results.learned_labels]
    mean_combined = statistics.fmean(combined)
    mean_voltage = statistics.fmean(voltage)
    mean_steps = statistics.fmean(steps)
    mean_feasible = statistics.fmean(feasible)
    if len(results.learned_labels) >= 2:
        seed_result = (
            f"Across {len(results.learned_labels)} independent seeds, "
            f"{results.algorithm_label} mean daily combined "
            f"cost is {mean_combined:.6g} with sample standard deviation "
            f"{statistics.stdev(combined):.6g}."
        )
    else:
        seed_result = (
            f"The selected {results.algorithm_label} run has mean daily combined cost "
            f"{mean_combined:.6g}."
        )
    neutral = summaries["Neutral"]
    regcontrol = summaries["RegControl"]
    boundary_shares = []
    for label in results.learned_labels:
        actions = [
            int(row[f"k{index}"])
            for row in results.hourly_rows[label]
            for index in range(1, results.action_count + 1)
        ]
        boundary_shares.append(sum(action in (0, N_TAPS - 1) for action in actions) / len(actions))
    saturation = ""
    if all(share >= 0.95 for share in boundary_shares):
        formatted = ", ".join(
            f"{label} {share:.1%}"
            for label, share in zip(results.learned_labels, boundary_shares, strict=True)
        )
        saturation = (
            f"The saved {results.algorithm_label} policies are saturated at the tap "
            f"limits: the boundary share of executed regulator-hour actions is {formatted}."
        )
    lines = [
        f"The {split} evaluation covers {len(results.evaluation_days)} days for "
        f"{len(results.learned_labels)} {results.algorithm_label} seed"
        f"{'s' if len(results.learned_labels) != 1 else ''}, beta "
        f"{results.settings['beta_text']}, and {budget}.",
        seed_result
        + " This is "
        + f"{_relation(mean_combined, neutral['Mean daily combined cost'])} Neutral "
        + f"({neutral['Mean daily combined cost']:.6g}) and "
        + f"{_relation(mean_combined, regcontrol['Mean daily combined cost'])} RegControl "
        + f"({regcontrol['Mean daily combined cost']:.6g}).",
        f"{results.algorithm_label} averages voltage cost {mean_voltage:.6g}, "
        f"{mean_steps:.3f} tap steps/day, "
        f"and {mean_feasible:.1f}% feasible hours. Neutral uses "
        f"{neutral['Mean tap steps/day']:.3f} steps/day with voltage cost "
        f"{neutral['Mean daily voltage cost']:.6g} and "
        f"{neutral['Feasible hours (%)']:.1f}% feasible hours; RegControl uses "
        f"{regcontrol['Mean tap steps/day']:.3f} steps/day with voltage cost "
        f"{regcontrol['Mean daily voltage cost']:.6g} and "
        f"{regcontrol['Feasible hours (%)']:.1f}% feasible hours.",
        f"Against RegControl, {results.algorithm_label}'s voltage cost is "
        f"{_relation(mean_voltage, regcontrol['Mean daily voltage cost'])} RegControl's, "
        f"its tap steps are {_relation(mean_steps, regcontrol['Mean tap steps/day'])} "
        f"RegControl's, and its feasible hours are "
        f"{_relation(mean_feasible, regcontrol['Feasible hours (%)'])} RegControl's.",
        saturation,
        f"These results are limited to the {split} split, this synthetic process, "
        "and the recorded training budget.",
    ]
    display(Markdown("\n\n".join(line for line in lines if line)))


def show_conclusions(results):
    """Display protocol-specific conclusions limited to the supplied results.

    Conclusions compare run-level learned means with references and report selected
    physical metrics. They retain the split, synthetic-process, seed, and recorded
    training-budget limits; they do not assert convergence or general performance.
    """
    if results.settings["protocol"] == "one_shot":
        return _show_conclusions_oneshot(results)
    return _show_conclusions_seq(results)
