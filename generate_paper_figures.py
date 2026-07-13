from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd


# Embed TrueType fonts in PDF output so figures do not rely on Type-3 glyphs.
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})


ROOT = Path(__file__).resolve().parent
FIGURES = ROOT.parent / "paper" / "figures"
GROUND_TRUTH_VISUALS = ROOT / "ground_truth" / "visuals"
COLORS = {"optional": "#187b75", "order": "#2f63d9", "rework": "#b45309"}


def _save(fig: plt.Figure, directory: Path, name: str, *, png: bool = False) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    if png:
        fig.savefig(directory / f"{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(directory / f"{name}.pdf", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _summary(experiment: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / "outputs" / experiment / "analysis" / "summary.csv")


def _results(experiment: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / "outputs" / experiment / "results.csv")


def _line(
    ax: plt.Axes,
    data: pd.DataFrame,
    metric: str,
    color: str,
    label: str,
    show_ci: bool = False,
    linestyle: str = "-",
    marker: str = "o",
    zorder: int = 2,
    error_color: str | None = None,
) -> None:
    x_values = data["variability"] * 100
    y_values = data[metric]
    if show_ci:
        base = metric.removesuffix("_median")
        low = data[f"{base}_ci_low"]
        high = data[f"{base}_ci_high"]
        errors = [y_values - low, high - y_values]
        ax.errorbar(
            x_values,
            y_values,
            yerr=errors,
            marker=marker,
            linestyle=linestyle,
            linewidth=2.1,
            capsize=3,
            color=color,
            ecolor=error_color or color,
            elinewidth=1.2,
            capthick=1.0,
            label=label,
            zorder=zorder,
        )
    else:
        ax.plot(
            x_values,
            y_values,
            marker=marker,
            linestyle=linestyle,
            linewidth=2.1,
            color=color,
            label=label,
            zorder=zorder,
        )
    ax.grid(True, alpha=0.25)
    ax.tick_params(axis="both", labelsize=9)


def _prepare_process_axis(ax: plt.Axes, title: str) -> None:
    ax.set_title(title, fontsize=12, weight="bold", pad=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def _box(ax: plt.Axes, x: float, y: float, text: str, color: str = "#334155") -> None:
    ax.add_patch(FancyBboxPatch((x - 0.055, y - 0.065), 0.11, 0.13, boxstyle="round,pad=0.01,rounding_size=0.01", linewidth=1.5, edgecolor=color, facecolor="#f8fafc"))
    ax.text(x, y, text, ha="center", va="center", fontsize=8.5, color="#1e293b")


def _arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = "#334155",
    dashed: bool = False,
    curve: float = 0.0,
    linestyle: str | None = None,
    linewidth: float = 1.4,
    shrinkA: float = 2.0,
    shrinkB: float = 2.0,
) -> None:
    line_style = linestyle if linestyle is not None else ("--" if dashed else "-")
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="->",
            mutation_scale=10,
            linewidth=linewidth,
            color=color,
            linestyle=line_style,
            connectionstyle=f"arc3,rad={curve}",
            shrinkA=shrinkA,
            shrinkB=shrinkB,
        )
    )


def _baseline(ax: plt.Axes, y: float = 0.5) -> list[tuple[float, float]]:
    labels = ["Register", "Check", "Assess", "Decide", "Notify", "Archive"]
    positions = [(0.09 + index * 0.165, y) for index in range(len(labels))]
    for position, label in zip(positions, labels, strict=True):
        _box(ax, *position, label)
    for start, end in zip(positions, positions[1:]):
        _arrow(ax, (start[0] + 0.055, start[1]), (end[0] - 0.055, end[1]))
    return positions


def draw_synthetic_process_overview() -> None:
    fig, ax = plt.subplots(figsize=(12, 5.3))
    _prepare_process_axis(ax, "Synthetic ground truth: baseline and isolated variability mechanisms")
    points = _baseline(ax, 0.55)
    docs, review = (0.34, 0.84), (0.51, 0.84)
    _box(ax, *docs, "Request\ndocuments", COLORS["optional"])
    _box(ax, *review, "Optional\nreview", COLORS["optional"])
    _arrow(ax, (points[1][0], 0.615), (docs[0] - 0.03, 0.775), COLORS["optional"], True)
    _arrow(ax, (docs[0] + 0.03, 0.775), (points[2][0], 0.615), COLORS["optional"], True)
    _arrow(ax, (points[2][0], 0.615), (review[0] - 0.03, 0.775), COLORS["optional"], True)
    _arrow(ax, (review[0] + 0.03, 0.775), (points[3][0], 0.615), COLORS["optional"], True)
    ax.text(0.87, 0.85, "optional work", color=COLORS["optional"], fontsize=9)
    _arrow(ax, (points[1][0], 0.49), (points[2][0], 0.49), COLORS["order"], curve=0.5, linestyle=":")
    _arrow(ax, (points[2][0], 0.49), (points[1][0], 0.49), COLORS["order"], curve=0.5, linestyle=":")
    ax.text(0.15, 0.24, "alternative order: Check and Assess may swap", color=COLORS["order"], fontsize=9)
    review_rework, rework = (0.42, 0.10), (0.55, 0.10)
    _box(ax, *review_rework, "Manual\nreview", COLORS["rework"])
    _box(ax, *rework, "Rework", COLORS["rework"])
    _arrow(ax, (points[2][0], 0.485), (review_rework[0] - 0.04, 0.165), COLORS["rework"], curve=0.2, linestyle="-.")
    _arrow(ax, (review_rework[0] + 0.055, 0.10), (rework[0] - 0.055, 0.10), COLORS["rework"], linestyle="-.")
    _arrow(ax, (rework[0] + 0.055, 0.10), (points[2][0] + 0.045, 0.485), COLORS["rework"], curve=-0.35, linestyle="-.")
    ax.text(0.61, 0.18, "Assess -> Manual review -> Rework -> Assess", color=COLORS["rework"], fontsize=9)
    _save(fig, FIGURES, "synthetic_process_overview")


def draw_ground_truth_visuals() -> None:
    descriptions = {
        "baseline": "one valid happy path",
        "optional": "two independent optional activities",
        "order": "two valid orders",
        "rework": "one-time repeated assessment path",
        "mixed": "combined variability; exact language in valid_variants.csv",
    }
    for scenario, description in descriptions.items():
        fig, ax = plt.subplots(figsize=(9, 2.8))
        _prepare_process_axis(ax, f"Ground truth - {scenario}: {description}")
        points = _baseline(ax, 0.55)
        if scenario == "optional":
            for x, label, source, target in [(0.34, "Request\ndocuments", points[1], points[2]), (0.51, "Manual\nreview", points[2], points[3])]:
                _box(ax, x, 0.84, label, COLORS["optional"])
                _arrow(ax, (source[0], 0.615), (x - 0.03, 0.775), COLORS["optional"], True)
                _arrow(ax, (x + 0.03, 0.775), (target[0], 0.615), COLORS["optional"], True)
        elif scenario == "order":
            labels = ["Register", "Assess", "Check", "Decide", "Notify", "Archive"]
            positions = [(0.09 + index * 0.165, 0.21) for index in range(len(labels))]
            for position, label in zip(positions, labels, strict=True):
                _box(ax, *position, label, COLORS["order"])
            for start, end in zip(positions, positions[1:]):
                _arrow(ax, (start[0] + 0.055, start[1]), (end[0] - 0.055, end[1]), COLORS["order"])
        elif scenario == "rework":
            positions = [(0.42, 0.18), (0.55, 0.18), (0.68, 0.18)]
            for position, label in zip(positions, ["Manual\nreview", "Rework", "Assess"], strict=True):
                _box(ax, *position, label, COLORS["rework"])
            _arrow(ax, (points[2][0], 0.485), (0.38, 0.245), COLORS["rework"], curve=0.2)
            _arrow(ax, (0.475, 0.18), (0.495, 0.18), COLORS["rework"])
            _arrow(ax, (0.605, 0.18), (0.625, 0.18), COLORS["rework"])
            _arrow(ax, (0.735, 0.18), (points[3][0], 0.485), COLORS["rework"], curve=-0.2)
        elif scenario == "mixed":
            ax.text(0.5, 0.18, "Optional work + alternative order + one-time rework", ha="center", fontsize=10)
        _save(fig, GROUND_TRUTH_VISUALS, f"{scenario}_overview", png=True)


def _three_across(name: str, metrics: list[tuple[str, str, str, str]], title: str) -> None:
    summary = _summary("main")
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 4.2), sharex=True)
    fig.suptitle(title, fontsize=16, y=0.995)
    labels = {"optional": "Optional activities", "order": "Alternative ordering", "rework": "Rework"}
    for column, (metric, ylabel, color, panel_label) in enumerate(metrics):
        scenario = list(labels)[column]
        frame = summary[summary["scenario"] == scenario].sort_values("variability")
        axes[column].set_title(f"{labels[scenario]}\n{panel_label}", fontsize=10.5, linespacing=1.25)
        _line(axes[column], frame, metric, color, ylabel)
        axes[column].set_ylabel(ylabel, fontsize=11)
        axes[column].set_xlabel("p (%)", fontsize=10)
        axes[column].tick_params(axis="both", labelsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93), w_pad=1.8)
    _save(fig, FIGURES, name)


def draw_structural_figures() -> None:
    _three_across("raw_structural_metrics", [
        ("imperative_bpmn_nodes_median", "BPMN nodes", "#2f7db7", "Imperative: BPMN nodes"),
        ("imperative_control_flow_complexity_median", "CFC", "#e07a1f", "Imperative: CFC"),
        ("declarative_declare_constraints_median", "Canonical Declare constraints", "#2d9d47", "Declarative: constraints"),
    ], "Representative direct metrics by variability mechanism")
    _three_across("activity_adjusted_structural_metrics", [
        ("imperative_bpmn_structural_overhead_median", "Non-task BPMN nodes", "#2f7db7", "Imperative: non-task nodes"),
        ("declarative_declare_constraints_per_observed_activity_median", "Canonical constraints / activity", "#2d9d47", "Declarative: constraints / activity"),
        ("declarative_declare_constraint_density_median", "Declare density", "#7c3aed", "Declarative: density"),
    ], "Representative activity-adjusted metrics by variability mechanism")


def draw_structural_summary() -> None:
    summary = _summary("main")
    scenarios = ["optional", "order", "rework"]
    labels = {
        "optional": "Optional activities",
        "order": "Alternative ordering",
        "rework": "Rework",
    }
    structural = (
        "imperative_bpmn_nodes_median",
        "BPMN nodes",
        "#2f7db7",
    )
    declarative = (
        "declarative_declare_constraints_median",
        "Canonical Declare constraints",
        "Declarative: constraints",
        "#2d9d47",
    )
    fig, axes = plt.subplots(3, 3, figsize=(10.4, 6.1), sharex="col", sharey="row")
    for column, scenario in enumerate(scenarios):
        frame = summary[summary["scenario"] == scenario].sort_values("variability")
        metric, ylabel, color = structural
        _line(axes[0, column], frame, metric, color, ylabel)
        axes[0, column].set_title(labels[scenario], fontsize=11, weight="bold", pad=8)

        metric, ylabel, panel_label, color = declarative
        _line(axes[1, column], frame, metric, color, ylabel)

        f1_series = [
            ("imperative_test_f1_median", "Inductive Miner", "#475569", "--", "s"),
            ("declarative_test_f1_median", "Declare semantics", "#d12f4f", "-", "D"),
            ("declarative_closed_world_test_f1_median", "Declare + closed-world", "#7c3aed", "-.", "^"),
        ]
        for metric, label, color, linestyle, marker in f1_series:
            axes[2, column].plot(
                frame["variability"] * 100,
                frame[metric],
                marker=marker,
                linestyle=linestyle,
                linewidth=1.8,
                markersize=4,
                color=color,
                label=label,
            )
        axes[2, column].set_title("Shared behavior: $F1_{\\mathrm{cls}}$", fontsize=10.5, pad=6)
        axes[2, column].set_ylim(0.55, 1.03)
        axes[2, column].set_xlabel("p (%)", fontsize=9)
        for row in range(3):
            axes[row, column].tick_params(axis="both", labelsize=8)

    axes[0, 0].set_ylabel("BPMN nodes", fontsize=9)
    axes[1, 0].set_ylabel("Declare constraints", fontsize=9)
    axes[2, 0].set_ylabel("$F1_{\\mathrm{cls}}$", fontsize=9)
    handles, legend_labels = axes[2, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.suptitle("Primary structural and behavioral results", fontsize=14, y=0.995)
    fig.text(0.012, 0.68, "Imperative", rotation=90, va="center", fontsize=9.5, weight="bold")
    fig.text(0.012, 0.40, "Declarative", rotation=90, va="center", fontsize=9.5, weight="bold")
    fig.text(0.012, 0.14, "Behavior", rotation=90, va="center", fontsize=9.5, weight="bold")
    fig.tight_layout(rect=(0.045, 0.10, 1, 0.95), h_pad=1.15, w_pad=1.35)
    _save(fig, FIGURES, "structural_summary")


def _read_bpmn_graph(path: Path) -> tuple[dict[str, str], list[tuple[str, str]]]:
    root = ET.parse(path).getroot()
    nodes: dict[str, str] = {}
    for element in root.iter():
        tag = element.tag.rsplit("}", 1)[-1]
        element_id = element.attrib.get("id")
        if not element_id:
            continue
        if tag == "task":
            nodes[element_id] = element.attrib.get("name", "activity")
        elif tag == "startEvent":
            nodes[element_id] = "start"
        elif tag == "endEvent":
            nodes[element_id] = "end"
        elif tag.endswith("Gateway"):
            direction = element.attrib.get("gatewayDirection", "")
            nodes[element_id] = "XOR split" if direction == "Diverging" else "XOR join"
    edges = []
    for element in root.iter():
        if element.tag.rsplit("}", 1)[-1] != "sequenceFlow":
            continue
        source = element.attrib.get("sourceRef")
        target = element.attrib.get("targetRef")
        if source in nodes and target in nodes:
            edges.append((source, target))
    return nodes, edges


def _constraint_record(records: list[dict], template: str, parameters: tuple[str, ...]) -> dict:
    for record in records:
        if record["template"] == template and tuple(record["parameters"]) == parameters:
            return record
    raise KeyError((template, parameters))


def _model_box(
    ax: plt.Axes,
    x: float,
    y: float,
    text: str,
    color: str = "#334155",
    width: float = 0.11,
    fontsize: float = 8.0,
) -> None:
    label = "Manual\nreview" if text == "Manual review" else text
    ax.add_patch(
        FancyBboxPatch(
            (x - width / 2, y - 0.052),
            width,
            0.104,
            boxstyle="round,pad=0.01,rounding_size=0.01",
            linewidth=1.5,
            edgecolor=color,
            facecolor="#f8fafc",
        )
    )
    ax.text(x, y, label, ha="center", va="center", fontsize=fontsize, color="#1e293b", linespacing=0.9)


def draw_representative_discovered_models() -> None:
    """Show a readable, simplified view derived from two exported models."""
    base = ROOT / "outputs" / "main" / "artifacts"
    seed_low = 1001
    seed_high = 1004
    low_path = base / "rework_p0p05_seed1001_im0p00_ds0p05_dc0p95"
    high_path = base / "rework_p0p05_seed1004_im0p00_ds0p05_dc0p95"
    nodes, edges = _read_bpmn_graph(low_path / "imperative_model.bpmn")
    low_records = json.loads((low_path / "declare_model.json").read_text())
    high_records = json.loads((high_path / "declare_model.json").read_text())

    fig = plt.figure(figsize=(10.6, 6.2))
    grid = fig.add_gridspec(3, 1, height_ratios=[1.10, 0.78, 0.58], hspace=0.28)
    bpmn_ax = fig.add_subplot(grid[0, 0])
    declare_ax = fig.add_subplot(grid[1, 0])
    explanation_ax = fig.add_subplot(grid[2, 0])

    bpmn_ax.set_title("Exported BPMN, rework $p=0.05$, seed 1001", fontsize=11, weight="bold", pad=8)
    bpmn_ax.set_xlim(-0.03, 1.03)
    bpmn_ax.set_ylim(0, 1)
    bpmn_ax.axis("off")
    positions = {
        "start": (0.03, 0.50),
        "Register": (0.13, 0.50),
        "Check": (0.26, 0.50),
        "Assess": (0.39, 0.50),
        "XOR split": (0.52, 0.50),
        "Decide": (0.65, 0.50),
        "Notify": (0.77, 0.50),
        "Archive": (0.89, 0.50),
        "end": (0.98, 0.50),
        "Manual review": (0.59, 0.80),
        "Rework": (0.75, 0.80),
        "XOR join": (0.52, 0.22),
    }
    for source_id, target_id in edges:
        if nodes[source_id] not in {"Manual review", "Rework", "XOR join"} and nodes[target_id] not in {"Manual review", "Rework", "XOR join"}:
            continue
        source = positions[nodes[source_id]]
        target = positions[nodes[target_id]]
        loop_edge = nodes[source_id] in {"Manual review", "Rework", "XOR join"} or nodes[target_id] in {"Manual review", "Rework", "XOR join"}
        _arrow(
            bpmn_ax,
            source,
            target,
            "#b45309" if loop_edge else "#475569",
            curve=0.12 if loop_edge else 0.0,
            linewidth=1.9 if loop_edge else 1.7,
            shrinkA=0,
            shrinkB=0,
        )
    for label, position in positions.items():
        if label not in nodes.values():
            continue
        if label in {"XOR split", "XOR join"}:
            bpmn_ax.scatter(*position, s=180, marker="D", color="#f59e0b", edgecolor="#92400e", zorder=3)
            bpmn_ax.text(*position, "X", ha="center", va="center", fontsize=8, weight="bold", zorder=4)
        elif label in {"start", "end"}:
            bpmn_ax.scatter(*position, s=90, color="#94a3b8", edgecolor="#334155", zorder=3)
        else:
            _model_box(
                bpmn_ax,
                *position,
                label,
                "#b45309" if label in {"Manual review", "Rework"} else "#475569",
                width=0.11 if label == "Manual review" else 0.08,
                fontsize=7.2 if label == "Manual review" else 7.6,
            )
    main_flow = ["start", "Register", "Check", "Assess", "XOR split", "Decide", "Notify", "Archive", "end"]
    half_width = {"start": 0.022, "end": 0.022, "XOR split": 0.035, "Register": 0.04, "Check": 0.04, "Assess": 0.04, "Decide": 0.04, "Notify": 0.04, "Archive": 0.04}
    for source_label, target_label in zip(main_flow, main_flow[1:]):
        y = positions[source_label][1]
        start = (positions[source_label][0] + half_width[source_label], y)
        end = (positions[target_label][0] - half_width[target_label], y)
        _arrow(bpmn_ax, start, end, "#475569", linewidth=2.0, shrinkA=0, shrinkB=0)
    bpmn_ax.text(0.52, 0.05, "Simplified layout from the exported BPMN graph", ha="center", fontsize=8, color="#64748b")

    declare_ax.set_title("Selected exported Declare records", fontsize=11, weight="bold", pad=8)
    declare_ax.set_xlim(0, 1)
    declare_ax.set_ylim(0, 1)
    declare_ax.axis("off")
    low_exact = _constraint_record(low_records, "exactly_one", ("Assess",))
    _constraint_record(low_records, "succession", ("Check", "Assess"))
    _constraint_record(high_records, "coexistence", ("Manual review", "Rework"))
    _constraint_record(high_records, "succession", ("Manual review", "Rework"))
    _constraint_record(high_records, "response", ("Rework", "Assess"))
    declare_ax.add_patch(FancyBboxPatch((0.01, 0.08), 0.47, 0.80, boxstyle="round,pad=0.012,rounding_size=0.01", linewidth=0.8, edgecolor="#fecaca", facecolor="#fff7f7"))
    declare_ax.add_patch(FancyBboxPatch((0.52, 0.08), 0.47, 0.80, boxstyle="round,pad=0.012,rounding_size=0.01", linewidth=0.8, edgecolor="#bbf7d0", facecolor="#f0fdf4"))
    declare_ax.text(0.04, 0.72, "Seed 1001: 37/67 constraints", fontsize=9.2, weight="bold", color="#7f1d1d")
    declare_ax.text(0.06, 0.50, f"exactly_one(Assess)  c={low_exact['confidence_ratio']:.3f}", fontsize=8.5, family="monospace", color="#b91c1c")
    declare_ax.text(0.06, 0.31, "succession(Check, Assess)", fontsize=8.5, family="monospace", color="#475569")
    declare_ax.text(0.55, 0.72, "Seed 1004: 64/96 constraints", fontsize=9.2, weight="bold", color="#166534")
    declare_ax.text(0.57, 0.50, "coexistence(Manual review, Rework)", fontsize=8.1, family="monospace", color="#15803d")
    declare_ax.text(0.57, 0.31, "succession(Manual review, Rework)", fontsize=8.1, family="monospace", color="#15803d")
    declare_ax.text(0.57, 0.15, "response(Rework, Assess)", fontsize=8.1, family="monospace", color="#15803d")

    explanation_ax.set_xlim(0, 1)
    explanation_ax.set_ylim(0, 1)
    explanation_ax.axis("off")
    explanation_ax.add_patch(FancyBboxPatch((0.01, 0.04), 0.98, 0.92, boxstyle="round,pad=0.015,rounding_size=0.01", linewidth=1.0, edgecolor="#cbd5e1", facecolor="#f8fafc"))
    explanation_ax.text(0.03, 0.82, "Why the seeds behave differently", fontsize=10, weight="bold", color="#1e293b")
    explanation_ax.text(0.03, 0.60, "Seed 1001 retains exactly_one(Assess). The model therefore rejects a valid trace with a second Assess", fontsize=8.6, color="#334155")
    explanation_ax.text(0.03, 0.48, "and reaches F1_cls = 0.588. Seed 1004 omits that rule and retains rework-specific relations,", fontsize=8.6, color="#334155")
    explanation_ax.text(0.03, 0.36, "so it accepts the valid rework behavior and reaches F1_cls = 1.000.", fontsize=8.6, color="#334155")
    explanation_ax.text(0.03, 0.16, "Both panels are simplified displays derived from exported discovery models; the complete BPMN and Declare files remain in the artifact.", fontsize=8.0, color="#64748b")
    _save(fig, FIGURES, "representative_discovered_models")


def draw_rework_figures() -> None:
    main = _summary("main")
    main_seed_rows = _results("main")
    frame = main[main["scenario"] == "rework"].sort_values("variability")
    seed_frame = main_seed_rows[main_seed_rows["scenario"] == "rework"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    _line(
        axes[0],
        frame,
        "declarative_test_f1_median",
        "#d12f4f",
        "Declare model semantics",
        linestyle="-",
        marker="D",
        zorder=3,
    )
    _line(
        axes[0],
        frame,
        "declarative_closed_world_test_f1_median",
        "#7c3aed",
        "Declare + closed-world alphabet",
        linestyle="-.",
        marker="^",
        zorder=4,
    )
    for variability, group in seed_frame.groupby("variability"):
        offsets = np.linspace(-0.65, 0.65, len(group))
        axes[0].scatter(
            variability * 100 + offsets,
            group["declarative_test_f1"],
            s=17,
            color="#d12f4f",
            alpha=0.45,
            edgecolors="none",
            zorder=4,
        )
    axes[0].set_title("Balanced variant-coverage $F1_{\\mathrm{cls}}$", fontsize=13)
    axes[0].set_xlabel("rework p (%)", fontsize=11)
    axes[0].set_ylabel("$F1_{\\mathrm{cls}}$", fontsize=11)
    axes[0].tick_params(axis="both", labelsize=11)
    axes[0].annotate(
        "Both Declare variants reach 1.0 from p=10%",
        xy=(10, 1.0),
        xytext=(15, 0.91),
        fontsize=8.2,
        color="#334155",
        arrowprops={"arrowstyle": "->", "color": "#64748b", "linewidth": 1},
    )
    axes[0].legend(fontsize=8.5)
    _line(
        axes[1],
        frame,
        "declarative_declare_constraints_median",
        "#2d9d47",
        "Canonical constraints",
    )
    for variability, group in seed_frame.groupby("variability"):
        offsets = np.linspace(-0.65, 0.65, len(group))
        axes[1].scatter(
            variability * 100 + offsets,
            group["declarative_declare_constraints"],
            s=17,
            color="#2d9d47",
            alpha=0.45,
            edgecolors="none",
            zorder=4,
        )
    axes[1].set_title("Declare constraints near the discovery boundary", fontsize=13)
    axes[1].set_xlabel("rework p (%)", fontsize=11)
    axes[1].set_ylabel("Canonical Declare constraints", fontsize=11)
    axes[1].tick_params(axis="both", labelsize=11)
    handles, labels = axes[1].get_legend_handles_labels()
    axes[1].legend(handles, labels, fontsize=8.5)
    fig.tight_layout()
    _save(fig, FIGURES, "rework_threshold_effect")

    sensitivity = _results("rework_sensitivity")
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.8), sharey=True)
    colors = {0.03: "#2f7db7", 0.05: "#e07a1f", 0.07: "#2d9d47"}
    styles = {0.03: ("o", "-"), 0.05: ("s", "--"), 0.07: ("^", ":")}
    for ax, confidence in zip(axes, [0.90, 0.95, 0.99], strict=True):
        subset = sensitivity[sensitivity["declare_min_confidence_ratio"] == confidence]
        for support in colors:
            support_rows = subset[subset["declare_min_support_ratio"] == support]
            perfect_share = (
                support_rows.groupby("variability")["declarative_test_f1"]
                .apply(lambda values: 100 * float((values == 1.0).mean()))
                .reset_index(name="perfect_share")
                .sort_values("variability")
            )
            ax.plot(
                perfect_share["variability"] * 100,
                perfect_share["perfect_share"],
                marker=styles[support][0],
                linestyle=styles[support][1],
                linewidth=2.1,
                color=colors[support],
                label=f"support {support:.2f}",
            )
            ax.grid(True, alpha=0.25)
        ax.set_title(f"confidence {confidence:.2f}", fontsize=13)
        ax.set_xlabel("rework p (%)", fontsize=11)
        ax.tick_params(axis="both", labelsize=11)
    axes[0].set_ylabel("Seeds with $F1_{\\mathrm{cls}}=1$ (%)", fontsize=11)
    axes[0].set_ylim(-3, 103)
    axes[-1].legend(fontsize=8.5, loc="lower right")
    fig.suptitle("Seed-level recovery across support and confidence thresholds", fontsize=16, y=1.03)
    fig.tight_layout()
    _save(fig, FIGURES, "rework_support_confidence_sensitivity")


def draw_rework_sensitivity_heatmap() -> None:
    sensitivity = _results("rework_sensitivity")
    p_levels = [0.03, 0.05, 0.07, 0.10]
    supports = [0.03, 0.05, 0.07]
    confidences = [0.90, 0.95, 0.99]
    fig, axes = plt.subplots(1, 3, figsize=(8.8, 2.9), sharey=True)
    for axis, confidence in zip(axes, confidences, strict=True):
        matrix = []
        for support in supports:
            row = []
            for prevalence in p_levels:
                values = sensitivity[
                    (sensitivity["declare_min_confidence_ratio"] == confidence)
                    & (sensitivity["declare_min_support_ratio"] == support)
                    & (sensitivity["variability"] == prevalence)
                ]["declarative_test_f1"]
                row.append(100 * float((values == 1.0).mean()))
            matrix.append(row)
        axis.imshow(matrix, vmin=0, vmax=100, cmap="YlGn", aspect="auto")
        axis.set_title(f"confidence {confidence:.2f}", fontsize=9.5)
        axis.set_xticks(range(len(p_levels)), ["3", "5", "7", "10"], fontsize=8)
        axis.set_xlabel("rework p (%)", fontsize=8.5)
        axis.set_yticks(range(len(supports)), [".03", ".05", ".07"], fontsize=8)
        for row_index in range(len(supports)):
            for column_index in range(len(p_levels)):
                axis.text(column_index, row_index, f"{matrix[row_index][column_index]:.0f}", ha="center", va="center", fontsize=8, color="#0f172a")
    axes[0].set_ylabel("minimum support", fontsize=8.5)
    fig.suptitle("Perfect-seed share in the rework sensitivity grid", fontsize=11.5, y=1.02)
    fig.text(
        0.5,
        0.01,
        "Cells show the percentage of seeds with $F1_{\\mathrm{cls}}=1$",
        ha="center",
        fontsize=8.5,
        color="#475569",
    )
    fig.tight_layout()
    _save(fig, FIGURES, "rework_sensitivity_heatmap")


def draw_mixed_figure() -> None:
    summary = _summary("mixed").sort_values("variability")
    metrics = [
        ("train_variant_count_median", "Observed variants", "#64748b"),
        ("imperative_bpmn_structural_overhead_median", "BPMN non-task nodes", "#2f7db7"),
        ("imperative_control_flow_complexity_median", "CFC", "#e07a1f"),
        ("declarative_declare_constraints_per_observed_activity_median", "Canonical constraints / activity", "#2d9d47"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.4))
    for ax, (metric, title, color) in zip(axes.flat, metrics, strict=True):
        _line(ax, summary, metric, color, title)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("p (%)", fontsize=10)
        ax.tick_params(axis="both", labelsize=10)
    fig.suptitle("Mixed-variability supplementary scenario", fontsize=15, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, FIGURES, "mixed_structural_metrics")


def main() -> None:
    # Figure 1 is the manually prepared paper graphic in paper/figures/image.pdf.
    # The remaining figures are regenerated from the experiment outputs below.
    draw_ground_truth_visuals()
    draw_structural_figures()
    draw_structural_summary()
    draw_representative_discovered_models()
    draw_rework_figures()
    draw_rework_sensitivity_heatmap()
    draw_mixed_figure()


if __name__ == "__main__":
    main()
