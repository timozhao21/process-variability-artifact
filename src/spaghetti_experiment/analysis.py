from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASELINE_CHANGE_COLUMNS = [
    "imperative_bpmn_nodes",
    "imperative_bpmn_flows",
    "imperative_control_flow_complexity",
    "imperative_bpmn_structural_overhead",
    "imperative_bpmn_nodes_per_observed_activity",
    "declarative_declare_constraints",
    "declarative_declare_constraints_primitive_preferred",
    "declarative_declare_relation_records_raw",
    "declarative_declare_constraints_per_observed_activity",
    "declarative_declare_raw_records_per_observed_activity",
    "declarative_declare_constraint_density",
    "declarative_declare_template_count",
]

# These are the measures interpreted in the paper.  A compact long-form file
# makes their seed-to-seed variation inspectable without opening the very wide
# full summary table.
KEY_SEED_METRICS = [
    "train_variant_entropy_normalized",
    "imperative_bpmn_nodes",
    "imperative_bpmn_structural_overhead",
    "imperative_control_flow_complexity",
    "declarative_declare_constraints",
    "declarative_declare_constraints_primitive_preferred",
    "declarative_declare_relation_records_raw",
    "declarative_declare_constraints_per_observed_activity",
    "imperative_test_f1",
    "declarative_test_f1",
    "declarative_test_f1_leave_one_family_out_range",
    "declarative_closed_world_test_f1",
    "imperative_native_precision",
    "declarative_full_trace_conformance",
]


def analyze_results(input_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _remove_retired_index_outputs(output_dir)
    dataframe = pd.read_csv(input_path)
    analyzed = add_baseline_changes(dataframe)
    analyzed.to_csv(output_dir / "results_with_baselines.csv", index=False)
    summary = summarize_results(analyzed)
    summary.to_csv(output_dir / "summary.csv", index=False)
    summarize_key_seed_metrics(analyzed).to_csv(
        output_dir / "key_metric_seed_summary.csv", index=False
    )
    write_plots(summary, output_dir)


def _remove_retired_index_outputs(output_dir: Path) -> None:
    for filename in ("results_with_indices.csv", "crossovers.csv"):
        (output_dir / filename).unlink(missing_ok=True)
    for path in output_dir.glob("*_indices.png"):
        path.unlink()


def add_baseline_changes(dataframe: pd.DataFrame) -> pd.DataFrame:
    changed = dataframe.copy()
    group_columns = [
        "scenario",
        "seed",
        "inductive_noise_threshold",
        "declare_min_support_ratio",
        "declare_min_confidence_ratio",
    ]
    present = [column for column in BASELINE_CHANGE_COLUMNS if column in changed.columns]
    for column in present:
        absolute_column = f"{column}_change_from_p0"
        percent_column = f"{column}_percent_change_from_p0"
        changed[absolute_column] = 0.0
        changed[percent_column] = np.nan
        for _, group in changed.groupby(group_columns, dropna=False):
            baseline = group.loc[group["variability"] == 0.0, column]
            if baseline.empty:
                continue
            baseline_value = float(baseline.iloc[0])
            index = group.index
            changed.loc[index, absolute_column] = changed.loc[index, column] - baseline_value
            if baseline_value != 0:
                changed.loc[index, percent_column] = (
                    changed.loc[index, absolute_column] / baseline_value * 100
                )
    return changed


def summarize_results(dataframe: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "scenario",
        "variability",
        "inductive_noise_threshold",
        "declare_min_support_ratio",
        "declare_min_confidence_ratio",
    ]
    metric_columns = [
        column
        for column in dataframe.columns
        if column not in {*group_columns, "seed"} and pd.api.types.is_numeric_dtype(dataframe[column])
    ]
    rows: list[dict[str, float | str]] = []
    for keys, group in dataframe.groupby(group_columns, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row: dict[str, float | str] = dict(zip(group_columns, keys, strict=True))
        for column in metric_columns:
            values = group[column].dropna().astype(float)
            if values.empty:
                continue
            row[f"{column}_median"] = float(values.median())
            row[f"{column}_ci_low"], row[f"{column}_ci_high"] = _bootstrap_median_ci(values)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_key_seed_metrics(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Write descriptive seed variation for the paper's primary measures."""
    group_columns = [
        "scenario",
        "variability",
        "inductive_noise_threshold",
        "declare_min_support_ratio",
        "declare_min_confidence_ratio",
    ]
    rows: list[dict[str, float | int | str]] = []
    for keys, group in dataframe.groupby(group_columns, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        condition = dict(zip(group_columns, keys, strict=True))
        for metric in KEY_SEED_METRICS:
            if metric not in group.columns:
                continue
            values = group[metric].dropna().astype(float)
            if values.empty:
                continue
            ci_low, ci_high = _bootstrap_median_ci(values)
            rows.append(
                {
                    **condition,
                    "metric": metric,
                    "n_seeds": int(values.size),
                    "median": float(values.median()),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "median_ci_low": ci_low,
                    "median_ci_high": ci_high,
                }
            )
    return pd.DataFrame(rows)


def write_plots(summary: pd.DataFrame, output_dir: Path) -> None:
    group_columns = [
        "scenario",
        "inductive_noise_threshold",
        "declare_min_support_ratio",
        "declare_min_confidence_ratio",
    ]
    for keys, group in summary.groupby(group_columns, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        scenario, noise_threshold, support, confidence = keys
        suffix = _plot_suffix(str(scenario), noise_threshold, support, confidence)
        ordered = group.sort_values("variability")
        _line_plot(
            ordered,
            output_dir / f"{suffix}_quality.png",
            "Test F1",
            [
                ("imperative_test_f1_median", "Imperative"),
                ("declarative_test_f1_median", "Declare, model semantics"),
                (
                    "declarative_closed_world_test_f1_median",
                    "Declare + closed-world alphabet",
                ),
            ],
        )
        _line_plot(
            ordered,
            output_dir / f"{suffix}_complexity.png",
            "Direct structural metrics",
            [
                ("imperative_bpmn_nodes_median", "BPMN nodes"),
                ("imperative_control_flow_complexity_median", "CFC"),
                ("declarative_declare_constraints_median", "Declare constraints"),
            ],
        )
        _line_plot(
            ordered,
            output_dir / f"{suffix}_variability.png",
            "Event-log variability",
            [
                ("train_variant_count_median", "Variant count"),
                ("train_variant_entropy_median", "Variant entropy"),
                ("train_variant_entropy_normalized_median", "Normalized entropy"),
            ],
        )


def _line_plot(
    dataframe: pd.DataFrame,
    path: Path,
    ylabel: str,
    columns: list[tuple[str, str]],
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    plotted = False
    for column, label in columns:
        if column in dataframe.columns:
            ax.plot(dataframe["variability"], dataframe[column], marker="o", label=label)
            plotted = True
    ax.set_xlabel("Variability")
    ax.set_ylabel(ylabel)
    if plotted:
        ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
def _bootstrap_median_ci(series: pd.Series) -> tuple[float, float]:
    values = series.to_numpy(dtype=float)
    if len(values) <= 1:
        value = float(values[0]) if len(values) else 0.0
        return value, value
    rng = np.random.default_rng(seed=len(values) * 7919)
    samples = rng.choice(values, size=(10000, len(values)), replace=True)
    medians = np.median(samples, axis=1)
    low, high = np.quantile(medians, [0.025, 0.975])
    return float(low), float(high)


def _plot_suffix(scenario: str, noise_threshold: float, support: float, confidence: float) -> str:
    if scenario != "noise" and noise_threshold == 0.0:
        return scenario
    return (
        f"{scenario}_im{_number_id(noise_threshold)}"
        f"_ds{_number_id(support)}"
        f"_dc{_number_id(confidence)}"
    )


def _number_id(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")
