from spaghetti_experiment.metrics import (
    classification_metrics,
    declare_complexity,
    mutation_robustness_metrics,
    trace_variability,
)
from spaghetti_experiment.models import declarative_acceptance
from spaghetti_experiment.generator import traces_to_dataframe


def test_classification_metrics() -> None:
    metrics = classification_metrics(
        positive_predictions=[True, True, False, True],
        negative_predictions=[False, True, False, False],
    )
    assert metrics["tp"] == 3
    assert metrics["fn"] == 1
    assert metrics["fp"] == 1
    assert metrics["tn"] == 3
    assert metrics["accuracy"] == 0.75
    assert metrics["f1"] == 0.75


def test_declare_complexity_counts_constraints() -> None:
    model = {
        "init": {"A": {"support": 10, "confidence": 10}},
        "response": {
            ("A", "B"): {"support": 10, "confidence": 9},
            ("B", "C"): {"support": 8, "confidence": 8},
        },
    }
    metrics = declare_complexity(
        model=model,
        observed_activities={"A", "B", "C"},
        allowed_templates={"init", "response"},
        case_count=10,
    )
    assert metrics["declare_constraints"] == 3
    assert metrics["declare_unary_constraints"] == 1
    assert metrics["declare_binary_constraints"] == 2
    assert metrics["declare_graph_max_degree"] == 2


def test_declare_complexity_canonicalizes_symmetric_and_composite_records() -> None:
    model = {
        "responded_existence": {
            ("A", "B"): {"support": 10, "confidence": 10},
            ("B", "A"): {"support": 10, "confidence": 10},
        },
        "coexistence": {
            ("A", "B"): {"support": 10, "confidence": 10},
            ("B", "A"): {"support": 10, "confidence": 10},
        },
        "response": {("A", "B"): {"support": 10, "confidence": 10}},
        "precedence": {("A", "B"): {"support": 10, "confidence": 10}},
        "succession": {("A", "B"): {"support": 10, "confidence": 10}},
    }
    metrics = declare_complexity(
        model=model,
        observed_activities={"A", "B"},
        allowed_templates=set(model),
        case_count=10,
    )

    assert metrics["declare_relation_records_raw"] == 7
    assert metrics["declare_symmetric_deduplicated_constraints"] == 6
    assert metrics["declare_constraints"] == 2
    assert metrics["declare_constraints_primitive_preferred"] == 4
    assert metrics["declare_redundant_records_removed"] == 5


def test_declare_density_degree_and_overlap_use_explicit_candidate_space() -> None:
    model = {
        "init": {"A": {"support": 10, "confidence": 10}},
        "responded_existence": {("A", "B"): {"support": 10, "confidence": 10}},
        "response": {("A", "B"): {"support": 10, "confidence": 10}},
        "coexistence": {("C", "B"): {"support": 10, "confidence": 10}},
    }
    metrics = declare_complexity(
        model=model,
        observed_activities={"A", "B", "C"},
        allowed_templates={"init", "responded_existence", "response", "coexistence"},
        case_count=10,
    )

    # 1 unary * 3 activities + 2 directed * 3*2 ordered pairs
    # + 1 symmetric * 3 unordered pairs = 18 candidates; four are present.
    assert metrics["declare_constraints"] == 4
    assert metrics["declare_constraint_density"] == 4 / 18
    assert metrics["declare_graph_mean_degree"] == 4 / 3
    assert metrics["declare_graph_max_degree"] == 2
    # The pair A->B has two directed relations, so it contributes one overlap.
    assert metrics["declare_relation_overlap"] == 1


def test_mutation_robustness_reports_macro_and_leave_one_out_values() -> None:
    metrics = mutation_robustness_metrics(
        positive_predictions=[True, True, False],
        negative_predictions=[False, False, True, False],
        mutation_labels=["missing", "missing", "order", "order"],
    )
    assert metrics["mutation_family_count"] == 2
    assert metrics["mutation_macro_rejection_rate"] == 0.75
    assert 0 <= metrics["mutation_macro_balanced_accuracy"] <= 1
    assert metrics["f1_leave_one_family_out_min"] <= metrics["f1_leave_one_family_out_max"]


def test_trace_variability_reports_entropy() -> None:
    metrics = trace_variability(
        [
            ("A", "B", "C"),
            ("A", "B", "C"),
            ("A", "C", "B"),
            ("A", "D", "C"),
        ]
    )
    assert metrics["variant_count"] == 3
    assert metrics["variant_ratio"] == 0.75
    assert metrics["variant_entropy"] > 0
    assert 0 < metrics["variant_entropy_normalized"] <= 1


def test_composed_declare_template_semantics() -> None:
    model = {
        "responded_existence": {("A", "B"): {}},
        "succession": {("A", "B"): {}},
    }
    accepted = traces_to_dataframe([("A", "B")], "accepted")
    missing_response = traces_to_dataframe([("A",)], "missing")
    wrong_order = traces_to_dataframe([("B", "A")], "wrong")

    assert declarative_acceptance(accepted, model, {"A", "B"}) == [True]
    assert declarative_acceptance(missing_response, model, {"A", "B"}) == [False]
    assert declarative_acceptance(wrong_order, model, {"A", "B"}) == [False]
