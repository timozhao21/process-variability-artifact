from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import pm4py

from .generator import InvalidTrace, dataframe_to_traces
from .models import ImperativeModel


def export_condition_artifacts(
    output_dir: Path,
    training_log: pd.DataFrame,
    positive_log: pd.DataFrame,
    negative_log: pd.DataFrame,
    invalid_traces: Sequence[InvalidTrace],
    imperative_model: ImperativeModel,
    declarative_model: dict[str, dict[Any, dict[str, int]]],
    imperative_positive_predictions: Sequence[bool],
    imperative_negative_predictions: Sequence[bool],
    declarative_positive_predictions: Sequence[bool],
    declarative_negative_predictions: Sequence[bool],
    declarative_closed_positive_predictions: Sequence[bool],
    declarative_closed_negative_predictions: Sequence[bool],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_log(training_log, output_dir / "training_log")
    _write_log(positive_log, output_dir / "positive_test_log")
    _write_log(negative_log, output_dir / "negative_test_log")
    _write_invalid_traces(invalid_traces, output_dir / "invalid_mutations.csv")
    _write_behavioral_diagnostics(
        positive_log,
        negative_log,
        invalid_traces,
        imperative_positive_predictions,
        imperative_negative_predictions,
        declarative_positive_predictions,
        declarative_negative_predictions,
        declarative_closed_positive_predictions,
        declarative_closed_negative_predictions,
        output_dir / "behavioral_diagnostics.csv",
    )
    pm4py.write_pnml(
        imperative_model.net,
        imperative_model.initial_marking,
        imperative_model.final_marking,
        str(output_dir / "imperative_model.pnml"),
    )
    pm4py.write_bpmn(
        imperative_model.bpmn,
        str(output_dir / "imperative_model.bpmn"),
        auto_layout=False,
    )
    (output_dir / "declare_model.json").write_text(
        json.dumps(serialize_declare_model(declarative_model), indent=2),
        encoding="utf-8",
    )


def serialize_declare_model(
    model: dict[str, dict[Any, dict[str, int]]]
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for template in sorted(model):
        for parameters, stats in sorted(
            model[template].items(), key=lambda item: str(item[0])
        ):
            parameter_list = list(parameters) if isinstance(parameters, tuple) else [parameters]
            support = int(stats.get("support", stats.get("support_count", 0)))
            confidence = int(stats.get("confidence", stats.get("confidence_count", 0)))
            rows.append(
                {
                    "template": template,
                    "parameters": parameter_list,
                    "support_count": support,
                    "confidence_count": confidence,
                    "confidence_ratio": confidence / support if support else 0.0,
                }
            )
    return rows


def condition_id(
    scenario: str,
    variability: float,
    seed: int,
    noise_threshold: float,
    declare_support: float,
    declare_confidence: float,
) -> str:
    return "_".join(
        [
            scenario,
            f"p{_number_id(variability)}",
            f"seed{seed}",
            f"im{_number_id(noise_threshold)}",
            f"ds{_number_id(declare_support)}",
            f"dc{_number_id(declare_confidence)}",
        ]
    )


def _write_log(log: pd.DataFrame, stem: Path) -> None:
    log.to_csv(stem.with_suffix(".csv"), index=False)
    pm4py.write_xes(log, str(stem.with_suffix(".xes")), show_progress_bar=False)


def _write_invalid_traces(invalid_traces: Sequence[InvalidTrace], path: Path) -> None:
    rows = [
        {
            "case_index": index,
            "mutation": item.mutation,
            "trace": " -> ".join(item.events),
        }
        for index, item in enumerate(invalid_traces)
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_behavioral_diagnostics(
    positive_log: pd.DataFrame,
    negative_log: pd.DataFrame,
    invalid_traces: Sequence[InvalidTrace],
    imperative_positive_predictions: Sequence[bool],
    imperative_negative_predictions: Sequence[bool],
    declarative_positive_predictions: Sequence[bool],
    declarative_negative_predictions: Sequence[bool],
    declarative_closed_positive_predictions: Sequence[bool],
    declarative_closed_negative_predictions: Sequence[bool],
    path: Path,
) -> None:
    rows: list[dict[str, object]] = []
    positive_traces = dataframe_to_traces(positive_log)
    negative_traces = dataframe_to_traces(negative_log)
    for index, trace in enumerate(positive_traces):
        rows.append(
            {
                "case_type": "positive",
                "case_index": index,
                "mutation_family": "valid_variant",
                "trace": " -> ".join(trace),
                "imperative_accepts": imperative_positive_predictions[index],
                "declare_model_accepts": declarative_positive_predictions[index],
                "declare_closed_world_accepts": declarative_closed_positive_predictions[index],
            }
        )
    for index, (trace, invalid) in enumerate(
        zip(negative_traces, invalid_traces, strict=True)
    ):
        rows.append(
            {
                "case_type": "negative",
                "case_index": index,
                "mutation_family": invalid.mutation,
                "trace": " -> ".join(trace),
                "imperative_accepts": imperative_negative_predictions[index],
                "declare_model_accepts": declarative_negative_predictions[index],
                "declare_closed_world_accepts": declarative_closed_negative_predictions[index],
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def _number_id(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")
