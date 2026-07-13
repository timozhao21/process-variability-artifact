from __future__ import annotations

from dataclasses import dataclass
import hashlib
from itertools import cycle, islice
import random
from typing import Iterable, Sequence

import pandas as pd

Trace = tuple[str, ...]

HAPPY_PATH: Trace = ("Register", "Check", "Assess", "Decide", "Notify", "Archive")
OPTIONAL_DOCUMENTS = "Request documents"
OPTIONAL_REVIEW = "Optional review"
REWORK_REVIEW = "Manual review"
REWORK = "Rework"
NOISE_INSERTIONS: Trace = (OPTIONAL_DOCUMENTS, OPTIONAL_REVIEW, REWORK_REVIEW, REWORK, "Escalate")


@dataclass(frozen=True)
class InvalidTrace:
    events: Trace
    mutation: str


def make_rng(seed: int, *parts: object) -> random.Random:
    material = "|".join([str(seed), *(str(part) for part in parts)])
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def generate_valid_traces(
    scenario: str, variability: float, count: int, rng: random.Random
) -> list[Trace]:
    return [_generate_trace(scenario, variability, rng) for _ in range(count)]


def generate_balanced_positive_traces(
    scenario: str, variability: float, count: int, rng: random.Random
) -> list[Trace]:
    variants = list(valid_variants(scenario, variability))
    if not variants:
        raise ValueError(f"No variants defined for scenario {scenario!r}.")
    traces = list(islice(cycle(variants), count))
    rng.shuffle(traces)
    return traces


def valid_variants(scenario: str, variability: float) -> tuple[Trace, ...]:
    if scenario in {"baseline", "noise"} or variability <= 0:
        return (HAPPY_PATH,)
    if scenario == "optional":
        return (
            HAPPY_PATH,
            ("Register", "Check", OPTIONAL_DOCUMENTS, "Assess", "Decide", "Notify", "Archive"),
            ("Register", "Check", "Assess", OPTIONAL_REVIEW, "Decide", "Notify", "Archive"),
            (
                "Register",
                "Check",
                OPTIONAL_DOCUMENTS,
                "Assess",
                OPTIONAL_REVIEW,
                "Decide",
                "Notify",
                "Archive",
            ),
        )
    if scenario == "order":
        return (HAPPY_PATH, ("Register", "Assess", "Check", "Decide", "Notify", "Archive"))
    if scenario == "rework":
        return (
            HAPPY_PATH,
            (
                "Register",
                "Check",
                "Assess",
                REWORK_REVIEW,
                REWORK,
                "Assess",
                "Decide",
                "Notify",
                "Archive",
            ),
        )
    if scenario == "mixed":
        return _mixed_variants()
    raise ValueError(f"Unknown scenario: {scenario}")


def is_valid_trace(scenario: str, trace: Sequence[str]) -> bool:
    events = tuple(trace)
    if scenario in {"baseline", "noise"}:
        return events == HAPPY_PATH
    if scenario == "optional":
        return _is_valid_optional(events)
    if scenario == "order":
        return events in valid_variants("order", 1.0)
    if scenario == "rework":
        return events in valid_variants("rework", 1.0)
    if scenario == "mixed":
        return events in valid_variants("mixed", 1.0)
    raise ValueError(f"Unknown scenario: {scenario}")


def generate_invalid_traces(
    scenario: str,
    valid_traces: Sequence[Trace],
    count: int,
    rng: random.Random,
) -> list[InvalidTrace]:
    truth_scenario = "baseline" if scenario == "noise" else scenario
    base_traces = list(valid_traces) or [HAPPY_PATH]
    families: dict[str, list[Trace]] = {}
    for base in dict.fromkeys(tuple(trace) for trace in base_traces):
        for mutation_name, mutated in _mutation_candidates(truth_scenario, base):
            if is_valid_trace(truth_scenario, mutated):
                continue
            candidates = families.setdefault(mutation_name, [])
            if mutated not in candidates:
                candidates.append(mutated)
    if not families:
        raise RuntimeError("Could not generate invalid mutation families.")

    family_names = sorted(families)
    rng.shuffle(family_names)
    for candidates in families.values():
        rng.shuffle(candidates)
    family_positions = {name: 0 for name in family_names}
    invalid: list[InvalidTrace] = []
    for mutation_name in islice(cycle(family_names), count):
        candidates = families[mutation_name]
        position = family_positions[mutation_name]
        invalid.append(
            InvalidTrace(
                events=candidates[position % len(candidates)],
                mutation=mutation_name,
            )
        )
        family_positions[mutation_name] += 1
    rng.shuffle(invalid)
    return invalid


def traces_to_dataframe(traces: Iterable[Sequence[str]], case_prefix: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    start = pd.Timestamp("2025-01-01 08:00:00")
    for case_index, trace in enumerate(traces):
        case_id = f"{case_prefix}_{case_index:05d}"
        for event_index, event in enumerate(trace):
            rows.append(
                {
                    "case:concept:name": case_id,
                    "concept:name": event,
                    "time:timestamp": start
                    + pd.Timedelta(days=case_index, minutes=event_index),
                    "event_index": event_index,
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "case:concept:name",
            "concept:name",
            "time:timestamp",
            "event_index",
        ],
    )


def dataframe_to_traces(dataframe: pd.DataFrame) -> list[Trace]:
    if dataframe.empty:
        return []
    ordered = dataframe.sort_values(["case:concept:name", "event_index", "time:timestamp"])
    traces: list[Trace] = []
    for _, group in ordered.groupby("case:concept:name", sort=False):
        traces.append(tuple(str(item) for item in group["concept:name"].tolist()))
    return traces


def _generate_trace(scenario: str, variability: float, rng: random.Random) -> Trace:
    if scenario == "baseline":
        return HAPPY_PATH
    if scenario == "noise":
        trace = HAPPY_PATH
        if rng.random() < variability:
            _, trace = rng.choice(_noise_mutations(trace, rng))
        return trace
    if scenario == "optional":
        trace = ["Register", "Check"]
        if rng.random() < variability:
            trace.append(OPTIONAL_DOCUMENTS)
        trace.append("Assess")
        if rng.random() < variability:
            trace.append(OPTIONAL_REVIEW)
        trace.extend(["Decide", "Notify", "Archive"])
        return tuple(trace)
    if scenario == "order":
        if rng.random() < variability:
            return ("Register", "Assess", "Check", "Decide", "Notify", "Archive")
        return HAPPY_PATH
    if scenario == "rework":
        if rng.random() < variability:
            return (
                "Register",
                "Check",
                "Assess",
                REWORK_REVIEW,
                REWORK,
                "Assess",
                "Decide",
                "Notify",
                "Archive",
            )
        return HAPPY_PATH
    if scenario == "mixed":
        return _generate_mixed_trace(variability, rng)
    raise ValueError(f"Unknown scenario: {scenario}")


def _is_valid_optional(events: Trace) -> bool:
    variants = set(valid_variants("optional", 1.0))
    return events in variants


def _mutation_candidates(scenario: str, base: Trace) -> list[tuple[str, Trace]]:
    common = [
        ("missing_decide", _remove_first(base, "Decide")),
        ("register_not_first", tuple([*base[1:], base[0]]) if base else base),
        ("archive_before_notify", _swap_first(base, "Notify", "Archive")),
        ("duplicate_notify", _insert_after(base, "Notify", "Notify")),
        ("notify_before_decide", _swap_first(base, "Decide", "Notify")),
        ("duplicate_register", (base[0], *base) if base else ("Register",)),
    ]
    if scenario == "optional":
        return common + [
            (
                "documents_after_assess",
                ("Register", "Check", "Assess", OPTIONAL_DOCUMENTS, "Decide", "Notify", "Archive"),
            ),
            (
                "review_before_assess",
                ("Register", "Check", OPTIONAL_REVIEW, "Assess", "Decide", "Notify", "Archive"),
            ),
            (
                "optional_after_decide",
                ("Register", "Check", "Assess", "Decide", OPTIONAL_REVIEW, "Notify", "Archive"),
            ),
        ]
    if scenario == "order":
        return common + [
            ("missing_check", _remove_first(base, "Check")),
            ("duplicate_assess", _insert_after(base, "Assess", "Assess")),
            (
                "check_after_decide",
                ("Register", "Assess", "Decide", "Check", "Notify", "Archive"),
            ),
        ]
    if scenario == "rework":
        return common + [
            (
                "review_without_rework",
                ("Register", "Check", "Assess", REWORK_REVIEW, "Decide", "Notify", "Archive"),
            ),
            (
                "rework_without_review",
                ("Register", "Check", "Assess", REWORK, "Assess", "Decide", "Notify", "Archive"),
            ),
            (
                "rework_without_second_assess",
                ("Register", "Check", "Assess", REWORK_REVIEW, REWORK, "Decide", "Notify", "Archive"),
            ),
            (
                "wrong_rework_order",
                ("Register", "Check", "Assess", REWORK, REWORK_REVIEW, "Assess", "Decide", "Notify", "Archive"),
            ),
        ]
    if scenario == "mixed":
        return common + [
            (
                "documents_after_decide",
                ("Register", "Check", "Assess", "Decide", OPTIONAL_DOCUMENTS, "Notify", "Archive"),
            ),
            (
                "manual_review_before_assess",
                ("Register", REWORK_REVIEW, "Check", "Assess", "Decide", "Notify", "Archive"),
            ),
            (
                "check_after_decide",
                ("Register", "Assess", "Decide", "Check", "Notify", "Archive"),
            ),
            (
                "rework_without_review",
                ("Register", "Check", "Assess", REWORK, "Assess", "Decide", "Notify", "Archive"),
            ),
            (
                "rework_without_second_assess",
                ("Register", "Check", "Assess", REWORK_REVIEW, REWORK, "Decide", "Notify", "Archive"),
            ),
            (
                "wrong_rework_order",
                ("Register", "Check", "Assess", REWORK, REWORK_REVIEW, "Assess", "Decide", "Notify", "Archive"),
            ),
        ]
    return common + _noise_mutations(base, random.Random(0))


def _generate_mixed_trace(variability: float, rng: random.Random) -> Trace:
    swapped_order = rng.random() < variability
    include_documents = rng.random() < variability
    include_rework = rng.random() < variability
    include_optional_review = (not include_rework) and rng.random() < variability

    trace = ["Register"]
    if swapped_order:
        trace.append("Assess")
        if include_optional_review:
            trace.append(OPTIONAL_REVIEW)
        if include_rework:
            trace.extend([REWORK_REVIEW, REWORK, "Assess"])
        trace.append("Check")
        if include_documents:
            trace.append(OPTIONAL_DOCUMENTS)
    else:
        trace.append("Check")
        if include_documents:
            trace.append(OPTIONAL_DOCUMENTS)
        trace.append("Assess")
        if include_optional_review:
            trace.append(OPTIONAL_REVIEW)
        if include_rework:
            trace.extend([REWORK_REVIEW, REWORK, "Assess"])
    trace.extend(["Decide", "Notify", "Archive"])
    return tuple(trace)


def _mixed_variants() -> tuple[Trace, ...]:
    variants: list[Trace] = []
    for swapped_order in (False, True):
        for include_documents in (False, True):
            for include_rework in (False, True):
                review_options = (False,) if include_rework else (False, True)
                for include_optional_review in review_options:
                    trace = ["Register"]
                    if swapped_order:
                        trace.append("Assess")
                        if include_optional_review:
                            trace.append(OPTIONAL_REVIEW)
                        if include_rework:
                            trace.extend([REWORK_REVIEW, REWORK, "Assess"])
                        trace.append("Check")
                        if include_documents:
                            trace.append(OPTIONAL_DOCUMENTS)
                    else:
                        trace.append("Check")
                        if include_documents:
                            trace.append(OPTIONAL_DOCUMENTS)
                        trace.append("Assess")
                        if include_optional_review:
                            trace.append(OPTIONAL_REVIEW)
                        if include_rework:
                            trace.extend([REWORK_REVIEW, REWORK, "Assess"])
                    trace.extend(["Decide", "Notify", "Archive"])
                    variants.append(tuple(trace))
    return tuple(variants)


def _noise_mutations(trace: Trace, rng: random.Random) -> list[tuple[str, Trace]]:
    mutations: list[tuple[str, Trace]] = []
    if len(trace) > 1:
        delete_index = rng.randrange(len(trace))
        mutations.append(("noise_delete", tuple(item for i, item in enumerate(trace) if i != delete_index)))
        swap_index = rng.randrange(len(trace) - 1)
        swapped = list(trace)
        swapped[swap_index], swapped[swap_index + 1] = swapped[swap_index + 1], swapped[swap_index]
        mutations.append(("noise_swap_adjacent", tuple(swapped)))
    insert_index = rng.randrange(len(trace) + 1)
    insertion = rng.choice(NOISE_INSERTIONS)
    inserted = list(trace)
    inserted.insert(insert_index, insertion)
    mutations.append(("noise_insert", tuple(inserted)))
    return mutations


def _remove_first(trace: Trace, event: str) -> Trace:
    items = list(trace)
    if event in items:
        items.remove(event)
    return tuple(items)


def _insert_after(trace: Trace, anchor: str, event: str) -> Trace:
    items = list(trace)
    try:
        index = items.index(anchor) + 1
    except ValueError:
        index = len(items)
    items.insert(index, event)
    return tuple(items)


def _swap_first(trace: Trace, left: str, right: str) -> Trace:
    items = list(trace)
    try:
        left_index = items.index(left)
        right_index = items.index(right)
    except ValueError:
        return tuple(items)
    items[left_index], items[right_index] = items[right_index], items[left_index]
    return tuple(items)
