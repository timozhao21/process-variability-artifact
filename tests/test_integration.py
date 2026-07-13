from pathlib import Path

import pm4py
import pandas as pd

from spaghetti_experiment.generator import (
    generate_invalid_traces,
    generate_valid_traces,
    make_rng,
    traces_to_dataframe,
)
from spaghetti_experiment.models import (
    declarative_acceptance,
    discover_declarative,
    discover_imperative,
    imperative_acceptance,
)
from spaghetti_experiment.config import ExperimentConfig
from spaghetti_experiment.runner import run_experiment


def test_both_paradigms_run_on_the_same_log() -> None:
    pm4py.util.constants.SHOW_PROGRESS_BAR = False
    training = generate_valid_traces("order", 0.2, 30, make_rng(21, "train"))
    positives = generate_valid_traces("order", 0.2, 10, make_rng(21, "positive"))
    negatives = generate_invalid_traces(
        "order", positives, 10, make_rng(21, "negative")
    )
    training_log = traces_to_dataframe(training, "train")
    positive_log = traces_to_dataframe(positives, "positive")
    negative_log = traces_to_dataframe(
        [item.events for item in negatives], "negative"
    )

    imperative = discover_imperative(training_log, noise_threshold=0.0)
    declarative = discover_declarative(
        training_log,
        frozenset(
            {
                "init",
                "exactly_one",
                "responded_existence",
                "response",
                "precedence",
                "succession",
                "coexistence",
            }
        ),
        min_support_ratio=0.05,
        min_confidence_ratio=0.95,
    )

    assert len(imperative_acceptance(positive_log, imperative)) == 10
    assert len(imperative_acceptance(negative_log, imperative)) == 10
    assert len(declarative_acceptance(positive_log, declarative)) == 10
    assert len(declarative_acceptance(negative_log, declarative)) == 10
    assert {"responded_existence", "succession", "coexistence"} <= set(declarative)


def test_unknown_activities_are_rejected_by_both_models() -> None:
    pm4py.util.constants.SHOW_PROGRESS_BAR = False
    training_log = traces_to_dataframe([("A", "B")] * 10, "train")
    unknown_log = traces_to_dataframe([("A", "X", "B")], "test")
    imperative = discover_imperative(training_log, noise_threshold=0.0)
    declarative = discover_declarative(
        training_log,
        frozenset({"init", "exactly_one", "response", "precedence"}),
        min_support_ratio=0.05,
        min_confidence_ratio=0.95,
    )

    assert imperative_acceptance(unknown_log, imperative) == [False]
    assert declarative_acceptance(unknown_log, declarative) == [True]
    assert declarative_acceptance(unknown_log, declarative, {"A", "B"}) == [False]


def test_artifact_index_references_existing_condition_files(tmp_path: Path) -> None:
    pm4py.util.constants.SHOW_PROGRESS_BAR = False
    config = ExperimentConfig(
        name="traceability_test",
        scenarios=("order",),
        variability_levels=(0.2,),
        seeds=(7,),
        train_traces=20,
        positive_test_traces=10,
        negative_test_traces=10,
        inductive_noise_thresholds=(0.0,),
        declare_min_support_ratios=(0.05,),
        declare_min_confidence_ratios=(0.95,),
        declare_templates=frozenset(
            {
                "init",
                "exactly_one",
                "responded_existence",
                "response",
                "precedence",
                "succession",
                "coexistence",
            }
        ),
        compute_native_quality=False,
        export_artifacts=True,
        show_progress_bars=False,
    )
    run_experiment(config, tmp_path)
    experiment_dir = tmp_path / config.name
    index = pd.read_csv(experiment_dir / "artifact_index.csv")
    results = pd.read_csv(experiment_dir / "results.csv")

    assert {
        "declarative_declare_relation_records_raw",
        "declarative_declare_constraints",
        "declarative_declare_constraints_primitive_preferred",
        "declarative_test_f1",
        "declarative_test_f1_leave_one_family_out_range",
        "declarative_closed_world_test_f1",
    } <= set(results.columns)
    assert (
        results["declarative_declare_constraints"]
        <= results["declarative_declare_relation_records_raw"]
    ).all()

    for column in [column for column in index.columns if column.endswith(("_csv", "_xes", "_bpmn", "_pnml", "_json"))]:
        assert all((experiment_dir / relative).exists() for relative in index[column])
