from __future__ import annotations

from itertools import product
import json
import platform
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd
import pm4py

from .artifacts import condition_id, export_condition_artifacts
from .config import ExperimentConfig
from .generator import (
    dataframe_to_traces,
    generate_balanced_positive_traces,
    generate_invalid_traces,
    generate_valid_traces,
    make_rng,
    traces_to_dataframe,
)
from .metrics import (
    classification_metrics,
    declare_complexity,
    mutation_robustness_metrics,
    trace_variability,
)
from .models import (
    declarative_acceptance,
    declarative_trace_conformance,
    discover_declarative,
    discover_imperative,
    imperative_acceptance,
    imperative_complexity,
    imperative_native_quality,
)


def run_experiment(config: ExperimentConfig, output_root: Path) -> Path:
    pm4py.util.constants.SHOW_PROGRESS_BAR = config.show_progress_bars
    experiment_dir = output_root / config.name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    artifact_rows: list[dict[str, Any]] = []
    combinations = product(
        config.scenarios,
        config.variability_levels,
        config.seeds,
        config.inductive_noise_thresholds,
        config.declare_min_support_ratios,
        config.declare_min_confidence_ratios,
    )
    for (
        scenario,
        variability,
        seed,
        inductive_noise_threshold,
        declare_support,
        declare_confidence,
    ) in combinations:
        row, artifact_row = _run_condition(
            config,
            experiment_dir,
            scenario,
            variability,
            seed,
            inductive_noise_threshold,
            declare_support,
            declare_confidence,
        )
        rows.append(row)
        if artifact_row is not None:
            artifact_rows.append(artifact_row)

    results_path = experiment_dir / "results.csv"
    pd.DataFrame(rows).to_csv(results_path, index=False)
    if artifact_rows:
        pd.DataFrame(artifact_rows).to_csv(experiment_dir / "artifact_index.csv", index=False)
    _write_manifest(config, experiment_dir)
    return results_path


def _run_condition(
    config: ExperimentConfig,
    experiment_dir: Path,
    scenario: str,
    variability: float,
    seed: int,
    inductive_noise_threshold: float,
    declare_support: float,
    declare_confidence: float,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    started = perf_counter()
    training = generate_valid_traces(
        scenario,
        variability,
        config.train_traces,
        make_rng(seed, scenario, variability, "train"),
    )
    truth_scenario = "baseline" if scenario == "noise" else scenario
    positives = generate_balanced_positive_traces(
        truth_scenario,
        variability,
        config.positive_test_traces,
        make_rng(seed, scenario, variability, "positive"),
    )
    negatives = generate_invalid_traces(
        scenario,
        positives,
        config.negative_test_traces,
        make_rng(seed, scenario, variability, "negative"),
    )

    training_log = traces_to_dataframe(training, "train")
    positive_log = traces_to_dataframe(positives, "positive")
    negative_log = traces_to_dataframe([item.events for item in negatives], "negative")

    imperative_model = discover_imperative(training_log, inductive_noise_threshold)
    declarative_model = discover_declarative(
        training_log,
        config.declare_templates,
        declare_support,
        declare_confidence,
    )

    observed_activities = set(str(item) for item in training_log["concept:name"].unique())
    imperative_positive_predictions = imperative_acceptance(positive_log, imperative_model)
    imperative_negative_predictions = imperative_acceptance(negative_log, imperative_model)
    declarative_positive_predictions = declarative_acceptance(positive_log, declarative_model)
    declarative_negative_predictions = declarative_acceptance(negative_log, declarative_model)
    declarative_closed_positive_predictions = declarative_acceptance(
        positive_log, declarative_model, observed_activities
    )
    declarative_closed_negative_predictions = declarative_acceptance(
        negative_log, declarative_model, observed_activities
    )
    mutation_labels = [item.mutation for item in negatives]

    imperative_metrics = classification_metrics(
        imperative_positive_predictions,
        imperative_negative_predictions,
    )
    declarative_metrics = classification_metrics(
        declarative_positive_predictions,
        declarative_negative_predictions,
    )
    declarative_closed_world_metrics = classification_metrics(
        declarative_closed_positive_predictions,
        declarative_closed_negative_predictions,
    )

    training_traces = dataframe_to_traces(training_log)
    row: dict[str, Any] = {
        "scenario": scenario,
        "variability": variability,
        "seed": seed,
        "train_traces": config.train_traces,
        "positive_test_traces": config.positive_test_traces,
        "negative_test_traces": config.negative_test_traces,
        "inductive_noise_threshold": inductive_noise_threshold,
        "declare_min_support_ratio": declare_support,
        "declare_min_confidence_ratio": declare_confidence,
        "train_activity_count": len(observed_activities),
    }
    imperative_structure = imperative_complexity(imperative_model)
    imperative_structure.update(
        {
            "bpmn_structural_overhead": (
                imperative_structure["bpmn_nodes"] - imperative_structure["bpmn_tasks"]
            ),
            "bpmn_nodes_per_observed_activity": (
                imperative_structure["bpmn_nodes"] / len(observed_activities)
            ),
        }
    )
    declarative_structure = declare_complexity(
        declarative_model,
        observed_activities,
        config.declare_templates,
        config.train_traces,
    )
    declarative_structure["declare_constraints_per_observed_activity"] = (
        declarative_structure["declare_constraints"] / len(observed_activities)
    )
    declarative_structure["declare_raw_records_per_observed_activity"] = (
        declarative_structure["declare_relation_records_raw"] / len(observed_activities)
    )
    declarative_structure["declare_primitive_constraints_per_observed_activity"] = (
        declarative_structure["declare_constraints_primitive_preferred"]
        / len(observed_activities)
    )
    row.update(_prefix("train_", trace_variability(training_traces)))
    row.update(_prefix("imperative_", imperative_structure))
    row.update(_prefix("declarative_", declarative_structure))
    row.update(_prefix("imperative_test_", imperative_metrics))
    row.update(
        _prefix(
            "imperative_test_",
            mutation_robustness_metrics(
                imperative_positive_predictions,
                imperative_negative_predictions,
                mutation_labels,
            ),
        )
    )
    row.update(_prefix("declarative_test_", declarative_metrics))
    row.update(
        _prefix(
            "declarative_test_",
            mutation_robustness_metrics(
                declarative_positive_predictions,
                declarative_negative_predictions,
                mutation_labels,
            ),
        )
    )
    row.update(
        _prefix("declarative_closed_world_test_", declarative_closed_world_metrics)
    )
    row.update(
        _prefix(
            "declarative_closed_world_test_",
            mutation_robustness_metrics(
                declarative_closed_positive_predictions,
                declarative_closed_negative_predictions,
                mutation_labels,
            ),
        )
    )

    if config.compute_native_quality:
        row.update(_prefix("imperative_", imperative_native_quality(training_log, imperative_model)))
        row.update(
            _prefix(
                "declarative_",
                declarative_trace_conformance(training_log, declarative_model),
            )
        )
    else:
        row.update(
            {
                "imperative_native_fitness": None,
                "imperative_native_precision": None,
                "imperative_native_generalization": None,
                "imperative_native_simplicity": None,
                "declarative_full_trace_conformance": None,
                "declarative_mean_constraint_satisfaction": None,
            }
        )

    row["runtime_seconds"] = perf_counter() - started

    artifact_row = None
    if config.export_artifacts:
        condition = condition_id(
            scenario,
            variability,
            seed,
            inductive_noise_threshold,
            declare_support,
            declare_confidence,
        )
        artifact_dir = experiment_dir / "artifacts" / condition
        export_condition_artifacts(
            artifact_dir,
            training_log,
            positive_log,
            negative_log,
            negatives,
            imperative_model,
            declarative_model,
            imperative_positive_predictions,
            imperative_negative_predictions,
            declarative_positive_predictions,
            declarative_negative_predictions,
            declarative_closed_positive_predictions,
            declarative_closed_negative_predictions,
        )
        relative_dir = artifact_dir.relative_to(experiment_dir).as_posix()
        artifact_row = {
            "condition_id": condition,
            "scenario": scenario,
            "variability": variability,
            "seed": seed,
            "inductive_noise_threshold": inductive_noise_threshold,
            "declare_min_support_ratio": declare_support,
            "declare_min_confidence_ratio": declare_confidence,
            "training_log_csv": f"{relative_dir}/training_log.csv",
            "training_log_xes": f"{relative_dir}/training_log.xes",
            "positive_test_log_csv": f"{relative_dir}/positive_test_log.csv",
            "positive_test_log_xes": f"{relative_dir}/positive_test_log.xes",
            "negative_test_log_csv": f"{relative_dir}/negative_test_log.csv",
            "negative_test_log_xes": f"{relative_dir}/negative_test_log.xes",
            "invalid_mutations_csv": f"{relative_dir}/invalid_mutations.csv",
            "behavioral_diagnostics_csv": f"{relative_dir}/behavioral_diagnostics.csv",
            "imperative_model_bpmn": f"{relative_dir}/imperative_model.bpmn",
            "imperative_model_pnml": f"{relative_dir}/imperative_model.pnml",
            "declare_model_json": f"{relative_dir}/declare_model.json",
        }

    return row, artifact_row


def _prefix(prefix: str, values: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}{key}": value for key, value in values.items()}


def _write_manifest(config: ExperimentConfig, experiment_dir: Path) -> None:
    manifest = {
        "config": config.to_dict(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "pm4py": getattr(pm4py, "__version__", None),
            "pandas": pd.__version__,
        },
    }
    (experiment_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
