import csv
from collections import Counter
from pathlib import Path

from spaghetti_experiment.generator import (
    generate_balanced_positive_traces,
    generate_invalid_traces,
    generate_valid_traces,
    is_valid_trace,
    make_rng,
    traces_to_dataframe,
    valid_variants,
)


def test_generation_is_reproducible() -> None:
    first = generate_valid_traces("optional", 0.2, 20, make_rng(7, "test"))
    second = generate_valid_traces("optional", 0.2, 20, make_rng(7, "test"))
    assert first == second


def test_generated_traces_match_ground_truth() -> None:
    for scenario in ("optional", "order", "rework", "mixed"):
        traces = generate_valid_traces(scenario, 0.5, 100, make_rng(9, scenario))
        assert all(is_valid_trace(scenario, trace) for trace in traces)


def test_balanced_evaluation_covers_all_variants() -> None:
    optional = generate_balanced_positive_traces(
        "optional", 0.05, 40, make_rng(10, "balanced")
    )
    order = generate_balanced_positive_traces(
        "order", 0.05, 20, make_rng(10, "order")
    )
    rework = generate_balanced_positive_traces(
        "rework", 0.05, 20, make_rng(10, "rework")
    )
    mixed = generate_balanced_positive_traces(
        "mixed", 0.05, 60, make_rng(10, "mixed")
    )
    assert len(set(optional)) == 4
    assert len(set(order)) == 2
    assert len(set(rework)) == 2
    assert len(set(mixed)) == 12


def test_invalid_traces_are_guaranteed_invalid() -> None:
    for scenario in ("optional", "order", "rework", "mixed", "noise"):
        valid_scenario = scenario if scenario != "noise" else "baseline"
        valid = generate_valid_traces(valid_scenario, 0.3, 25, make_rng(12, scenario))
        invalid = generate_invalid_traces(
            scenario, valid, 50, make_rng(13, scenario)
        )
        truth = scenario if scenario != "noise" else "baseline"
        assert all(not is_valid_trace(truth, item.events) for item in invalid)


def test_invalid_mutation_families_are_balanced() -> None:
    valid = generate_balanced_positive_traces(
        "rework", 0.05, 40, make_rng(12, "balanced-invalid")
    )
    invalid = generate_invalid_traces(
        "rework", valid, 150, make_rng(13, "balanced-invalid")
    )
    counts = Counter(item.mutation for item in invalid)
    assert max(counts.values()) - min(counts.values()) <= 1


def test_dataframe_has_process_mining_columns() -> None:
    dataframe = traces_to_dataframe([("A", "B"), ("A", "C")], "case")
    assert {
        "case:concept:name",
        "concept:name",
        "time:timestamp",
        "event_index",
    } <= set(dataframe.columns)
    assert dataframe["case:concept:name"].nunique() == 2


def test_ground_truth_variant_table_matches_generator() -> None:
    table_path = Path(__file__).resolve().parents[1] / "ground_truth" / "valid_variants.csv"
    documented: dict[str, set[tuple[str, ...]]] = {}
    with table_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            documented.setdefault(row["scenario"], set()).add(
                tuple(row["trace"].split(" -> "))
            )

    for scenario in ("baseline", "optional", "order", "rework", "mixed"):
        assert documented[scenario] == set(valid_variants(scenario, 0.5))
