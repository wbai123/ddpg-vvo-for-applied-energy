"""Build feeder-independent notebook figures from aligned saved results."""

import matplotlib.pyplot as plt
import numpy as np

from feeder.taps import ratio_of_index


def tap_ratio_fields(indices):
    """Convert regulator-ordered positions to numbered physical p.u. ratios."""
    ratios = ratio_of_index(indices)
    return {f"Ratio {index}": float(value) for index, value in enumerate(ratios, start=1)}


def hourly_detail_rows(selected_hourly, method_labels, action_count):
    """Build a 24-hour executed-position, ratio, movement, and cost table."""
    rows = []
    for hour in range(24):
        for label in method_labels:
            source = selected_hourly[label][hour]
            indices = [int(source[f"k{index}"]) for index in range(1, action_count + 1)]
            row = {"Hour": hour, "Method": label}
            row.update((f"Index {index}", value) for index, value in enumerate(indices, 1))
            row.update(tap_ratio_fields(indices))
            row.update(
                {
                    "Tap steps": int(source["tap_steps"]),
                    "Voltage cost": float(source["voltage_cost"]),
                    "Switching cost": float(source["switching_cost"]),
                }
            )
            rows.append(row)
    return rows


def _coincident_caption(sequences, labels):
    """Name labels with identical executed indices on every regulator."""
    groups = []
    for label in labels:
        sequence = sequences[label]
        for group in groups:
            if sequences[group[0]] == sequence:
                group.append(label)
                break
        else:
            groups.append([label])
    return " ".join(
        f"Curves for {', '.join(group)} coincide." for group in groups if len(group) > 1
    )


def plot_cost_feasibility(summaries, labels, split, colors):
    """Plot static mean cost and feasibility with a zero-safe cost scale."""
    mean_costs = [summaries[label]["Mean hourly voltage cost"] for label in labels]
    positions = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=100)
    axes[0].bar(positions, mean_costs, color=[colors[label] for label in labels])
    scale = "log" if all(cost > 0 for cost in mean_costs) else "linear"
    if scale == "log":
        axes[0].set_yscale("log")
    axes[0].set_ylabel(f"Mean hourly voltage cost ({scale} scale)")
    axes[0].set_title(f"Mean voltage cost on {split}")
    axes[1].bar(
        positions,
        [summaries[label]["Feasible scenarios (%)"] for label in labels],
        color=[colors[label] for label in labels],
    )
    axes[1].set_ylim(0, 100)
    axes[1].set_ylabel("Feasible scenarios (%)")
    axes[1].set_title(f"Scenarios inside [0.95, 1.05] p.u. on {split}")
    for axis in axes:
        axis.set_xticks(positions, labels, rotation=30, ha="right")
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig, axes


def plot_static_taps(scenarios, result_rows, labels, colors, action_count):
    """Plot executed ratios against independently sorted static multipliers."""
    multipliers = np.asarray([float(scenario.factors[0]) for scenario in scenarios])
    order = np.argsort(multipliers, kind="stable")
    fig, axes = plt.subplots(
        1, action_count, figsize=(4 * action_count, 3.6), dpi=100, sharex=True, sharey=True
    )
    axes = np.atleast_1d(axes)
    indices_by_label = {
        label: tuple(
            tuple(int(row[f"k{regulator}"]) for row in result_rows[label])
            for regulator in range(1, action_count + 1)
        )
        for label in labels
    }
    caption = _coincident_caption(indices_by_label, labels)
    linestyles = ("-", "--", "-.", ":")
    for regulator, axis in enumerate(axes, start=1):
        axis.axhline(1.0, color="#b0b0b0", linewidth=0.7, zorder=0)
        for bound in (0.90, 1.10):
            axis.axhline(
                bound,
                color="#b0b0b0",
                linewidth=0.7,
                linestyle="--",
                zorder=0,
                label="legal bounds" if axis is axes[-1] and bound == 0.90 else None,
            )
        for label_index, label in enumerate(labels):
            indices = np.asarray(indices_by_label[label][regulator - 1])
            axis.plot(
                multipliers[order],
                ratio_of_index(indices)[order],
                color=colors[label],
                linewidth=0.7,
                alpha=0.75,
                linestyle=linestyles[label_index % len(linestyles)],
                label=label,
            )
        axis.set_title(f"Regulator {regulator}")
        axis.set_xlabel("Loading multiplier")
        axis.set_ylim(0.88, 1.12)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Executed ratio (p.u.)")
    axes[-1].legend(fontsize=7)
    if caption:
        fig.text(0.5, 0.01, caption, ha="center", color="#666666", fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 1) if caption else None)
    return fig, axes


def plot_day_ratios(day_id, multipliers, selected_hourly, labels, colors, action_count):
    """Plot one day's hourly multiplier and persistent executed tap ratios."""
    hours = np.arange(24)
    fig, axes = plt.subplots(
        action_count + 1,
        1,
        figsize=(10, 2.1 * (action_count + 1)),
        dpi=100,
        sharex=True,
        gridspec_kw={"height_ratios": [0.85, *([1] * action_count)]},
    )
    axes[0].plot(hours, multipliers, color="#4c4c4c", marker="o", linewidth=0.8)
    axes[0].set_title(f"Loading and executed regulator ratios on {day_id}")
    axes[0].set_ylabel("Loading\nmultiplier")
    axes[0].grid(alpha=0.25)
    indices_by_label = {
        label: tuple(
            tuple(int(row[f"k{regulator}"]) for row in selected_hourly[label])
            for regulator in range(1, action_count + 1)
        )
        for label in labels
    }
    caption = _coincident_caption(indices_by_label, labels)
    markers = ("o", "s", "^", "D", "v")
    for regulator, axis in enumerate(axes[1:], start=1):
        axis.axhline(1.0, color="#b0b0b0", linewidth=0.7, zorder=0)
        for bound in (0.90, 1.10):
            axis.axhline(
                bound,
                color="#b0b0b0",
                linewidth=0.7,
                linestyle="--",
                zorder=0,
                label="legal bounds" if regulator == 1 and bound == 0.90 else None,
            )
        for label_index, label in enumerate(labels):
            indices = indices_by_label[label][regulator - 1]
            axis.step(
                hours,
                ratio_of_index(indices),
                where="post",
                marker=markers[label_index % len(markers)],
                markersize=2.5,
                color=colors[label],
                label=label,
            )
        axis.set_ylabel(f"Regulator {regulator}\nratio (p.u.)")
        axis.set_ylim(0.88, 1.12)
        axis.grid(alpha=0.25)
    axes[-1].set_xlabel("Hour (count)")
    axes[-1].set_xticks(hours)
    axes[1].legend(fontsize=7, ncol=3)
    if caption:
        fig.text(0.5, 0.01, caption, ha="center", color="#666666", fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 1) if caption else None)
    return fig, axes
