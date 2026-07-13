from __future__ import annotations

from collections import Counter, defaultdict
import math
from typing import Any, Iterable


UNARY_TEMPLATES = frozenset({"init", "existence", "exactly_one", "absence"})
SYMMETRIC_TEMPLATES = frozenset({"coexistence", "noncoexistence"})


def classification_metrics(
    positive_predictions: Iterable[bool], negative_predictions: Iterable[bool]
) -> dict[str, float]:
    positives = list(positive_predictions)
    negatives = list(negative_predictions)
    tp = sum(1 for item in positives if item)
    fn = len(positives) - tp
    fp = sum(1 for item in negatives if item)
    tn = len(negatives) - fp
    total = tp + fn + fp + tn
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    return {
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "accuracy": _safe_div(tp + tn, total),
        "precision": precision,
        "recall": recall,
        "f1": _safe_div(2 * precision * recall, precision + recall),
        "false_acceptance_rate": _safe_div(fp, fp + tn),
        "false_rejection_rate": _safe_div(fn, tp + fn),
    }


def mutation_robustness_metrics(
    positive_predictions: Iterable[bool],
    negative_predictions: Iterable[bool],
    mutation_labels: Iterable[str],
) -> dict[str, float]:
    """Summarize sensitivity to the selected invalid-mutation families.

    The macro rejection rate gives every mutation family equal weight. The
    leave-one-family-out range shows how much F1 changes when any one family is
    removed from the controlled negative set.
    """
    positives = list(positive_predictions)
    negatives = list(negative_predictions)
    labels = list(mutation_labels)
    if len(negatives) != len(labels):
        raise ValueError("Each negative prediction needs one mutation label.")
    families = sorted(set(labels))
    if not families:
        return {
            "mutation_family_count": 0.0,
            "mutation_macro_rejection_rate": 0.0,
            "mutation_macro_balanced_accuracy": 0.0,
            "f1_leave_one_family_out_min": 0.0,
            "f1_leave_one_family_out_max": 0.0,
            "f1_leave_one_family_out_range": 0.0,
        }

    rejection_rates = []
    for family in families:
        family_predictions = [
            prediction
            for prediction, label in zip(negatives, labels, strict=True)
            if label == family
        ]
        rejection_rates.append(
            sum(not prediction for prediction in family_predictions)
            / len(family_predictions)
        )
    positive_recall = _safe_div(sum(positives), len(positives))
    macro_rejection = sum(rejection_rates) / len(rejection_rates)

    leave_one_out_f1 = []
    for omitted in families:
        retained = [
            prediction
            for prediction, label in zip(negatives, labels, strict=True)
            if label != omitted
        ]
        if retained:
            leave_one_out_f1.append(
                classification_metrics(positives, retained)["f1"]
            )
    if not leave_one_out_f1:
        leave_one_out_f1 = [classification_metrics(positives, negatives)["f1"]]
    minimum = min(leave_one_out_f1)
    maximum = max(leave_one_out_f1)
    return {
        "mutation_family_count": float(len(families)),
        "mutation_macro_rejection_rate": macro_rejection,
        "mutation_macro_balanced_accuracy": (positive_recall + macro_rejection) / 2,
        "f1_leave_one_family_out_min": minimum,
        "f1_leave_one_family_out_max": maximum,
        "f1_leave_one_family_out_range": maximum - minimum,
    }


def trace_variability(traces: Iterable[tuple[str, ...]]) -> dict[str, float]:
    trace_list = list(traces)
    case_count = len(trace_list)
    counts = Counter(trace_list)
    variant_count = len(counts)
    entropy = 0.0
    for count in counts.values():
        probability = count / case_count if case_count else 0.0
        if probability:
            entropy -= probability * math.log2(probability)
    max_entropy = math.log2(variant_count) if variant_count > 1 else 0.0
    lengths = [len(trace) for trace in trace_list]
    return {
        "variant_count": variant_count,
        "variant_ratio": _safe_div(variant_count, case_count),
        "variant_entropy": entropy,
        "variant_entropy_normalized": _safe_div(entropy, max_entropy),
        "mean_trace_length": _safe_div(sum(lengths), len(lengths)),
    }


def declare_complexity(
    model: dict[str, dict[Any, dict[str, float]]],
    observed_activities: set[str],
    allowed_templates: set[str] | frozenset[str],
    case_count: int,
) -> dict[str, float]:
    raw_constraints = list(_iter_constraints(model))
    symmetric_deduplicated = _deduplicate_symmetric_constraints(raw_constraints)
    constraints = _remove_composition_redundancy(symmetric_deduplicated)
    primitive_preferred = _remove_composite_redundancy(symmetric_deduplicated)
    unary = [item for item in constraints if item[0] in UNARY_TEMPLATES]
    binary = [item for item in constraints if item[0] not in UNARY_TEMPLATES]
    template_count = len({template for template, _, _ in constraints})

    constrained_activities: set[str] = set()
    graph: dict[str, set[str]] = defaultdict(set)
    pair_template_counter: Counter[tuple[str, str]] = Counter()
    support_ratios: list[float] = []
    confidence_ratios: list[float] = []

    for template, parameters, stats in constraints:
        parameter_tuple = _parameter_tuple(parameters)
        constrained_activities.update(parameter_tuple)
        support = float(stats.get("support", stats.get("support_count", 0)))
        confidence = float(stats.get("confidence", stats.get("confidence_count", 0)))
        if case_count:
            support_ratios.append(support / case_count)
        if support:
            confidence_ratios.append(confidence / support)
        if template not in UNARY_TEMPLATES and len(parameter_tuple) >= 2:
            left, right = parameter_tuple[:2]
            graph[left].add(right)
            graph[right].add(left)
            pair_template_counter[(left, right)] += 1

    for activity in observed_activities:
        graph.setdefault(activity, set())

    unary_template_count = len(set(allowed_templates) & UNARY_TEMPLATES)
    binary_templates = set(allowed_templates) - UNARY_TEMPLATES
    symmetric_template_count = len(binary_templates & SYMMETRIC_TEMPLATES)
    directed_template_count = len(binary_templates - SYMMETRIC_TEMPLATES)
    activity_count = len(observed_activities)
    directed_pairs = activity_count * max(activity_count - 1, 0)
    unordered_pairs = directed_pairs / 2
    potential = (
        unary_template_count * activity_count
        + directed_template_count * directed_pairs
        + symmetric_template_count * unordered_pairs
    )
    relation_overlap = sum(max(0, count - 1) for count in pair_template_counter.values())
    degrees = [len(neighbors) for neighbors in graph.values()]

    return {
        "declare_relation_records_raw": len(raw_constraints),
        "declare_symmetric_deduplicated_constraints": len(symmetric_deduplicated),
        "declare_constraints": len(constraints),
        "declare_constraints_primitive_preferred": len(primitive_preferred),
        "declare_redundant_records_removed": len(raw_constraints) - len(constraints),
        "declare_unary_constraints": len(unary),
        "declare_binary_constraints": len(binary),
        "declare_template_count": template_count,
        "declare_observed_activities": activity_count,
        "declare_constrained_activities": len(constrained_activities),
        "declare_unconstrained_activities": len(set(observed_activities) - constrained_activities),
        "declare_constraint_density": _safe_div(len(constraints), potential),
        "declare_graph_mean_degree": _safe_div(sum(degrees), len(degrees)),
        "declare_graph_max_degree": max(degrees, default=0),
        "declare_relation_overlap": relation_overlap,
        "declare_mean_support_ratio": _safe_div(sum(support_ratios), len(support_ratios)),
        "declare_mean_confidence_ratio": _safe_div(sum(confidence_ratios), len(confidence_ratios)),
    }


def _iter_constraints(model: dict[str, dict[Any, dict[str, float]]]):
    for template, parameter_map in model.items():
        for parameters, stats in parameter_map.items():
            yield template, parameters, stats


def _deduplicate_symmetric_constraints(
    constraints: list[tuple[str, Any, dict[str, float]]],
) -> list[tuple[str, tuple[str, ...], dict[str, float]]]:
    """Collapse reverse duplicates for symmetric Declare templates.

    PM4Py can expose both ``coexistence(A, B)`` and ``coexistence(B, A)``.
    They are two serialized relation records but one semantic constraint.
    """
    deduplicated: list[tuple[str, tuple[str, ...], dict[str, float]]] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for template, parameters, stats in constraints:
        parameter_tuple = _parameter_tuple(parameters)
        canonical_parameters = (
            tuple(sorted(parameter_tuple))
            if template in SYMMETRIC_TEMPLATES and len(parameter_tuple) >= 2
            else parameter_tuple
        )
        key = (template, canonical_parameters)
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append((template, canonical_parameters, stats))
    return deduplicated


def _remove_composition_redundancy(
    constraints: list[tuple[str, tuple[str, ...], dict[str, float]]],
) -> list[tuple[str, tuple[str, ...], dict[str, float]]]:
    """Prefer one composite relation over equivalent primitive records.

    ``succession(A, B)`` combines ``response(A, B)`` and
    ``precedence(A, B)``. ``coexistence(A, B)`` combines the two directed
    responded-existence relations. Keeping all records would make structural
    size depend on PM4Py's serialization rather than the displayed semantics.
    """
    keys = {(template, parameters) for template, parameters, _ in constraints}
    reduced: list[tuple[str, tuple[str, ...], dict[str, float]]] = []
    for template, parameters, stats in constraints:
        if template in {"response", "precedence"} and (
            "succession",
            parameters,
        ) in keys:
            continue
        if template == "responded_existence" and len(parameters) >= 2:
            unordered = tuple(sorted(parameters[:2]))
            if ("coexistence", unordered) in keys:
                continue
        reduced.append((template, parameters, stats))
    return reduced


def _remove_composite_redundancy(
    constraints: list[tuple[str, tuple[str, ...], dict[str, float]]],
) -> list[tuple[str, tuple[str, ...], dict[str, float]]]:
    """Alternative sensitivity policy that retains primitive relations.

    This is not used for trace acceptance. It provides a second structural count
    so conclusions can be checked against the opposite representation choice.
    """
    keys = {(template, parameters) for template, parameters, _ in constraints}
    reduced: list[tuple[str, tuple[str, ...], dict[str, float]]] = []
    for template, parameters, stats in constraints:
        if template == "succession" and (
            "response",
            parameters,
        ) in keys and ("precedence", parameters) in keys:
            continue
        if template == "coexistence" and len(parameters) >= 2:
            left, right = parameters[:2]
            if (
                "responded_existence",
                (left, right),
            ) in keys and ("responded_existence", (right, left)) in keys:
                continue
        reduced.append((template, parameters, stats))
    return reduced


def _parameter_tuple(parameters: Any) -> tuple[str, ...]:
    if isinstance(parameters, tuple):
        return tuple(str(item) for item in parameters)
    if isinstance(parameters, list):
        return tuple(str(item) for item in parameters)
    return (str(parameters),)


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0
