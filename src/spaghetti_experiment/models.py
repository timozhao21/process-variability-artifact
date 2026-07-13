from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import pm4py

from .generator import Trace, dataframe_to_traces


@dataclass(frozen=True)
class ImperativeModel:
    process_tree: Any
    net: Any
    initial_marking: Any
    final_marking: Any
    bpmn: Any


def discover_imperative(training_log: pd.DataFrame, noise_threshold: float = 0.0) -> ImperativeModel:
    process_tree = pm4py.discover_process_tree_inductive(
        training_log,
        noise_threshold=noise_threshold,
    )
    net, initial_marking, final_marking = pm4py.convert_to_petri_net(process_tree)
    bpmn = pm4py.convert_to_bpmn(process_tree)
    return ImperativeModel(
        process_tree=process_tree,
        net=net,
        initial_marking=initial_marking,
        final_marking=final_marking,
        bpmn=bpmn,
    )


def discover_declarative(
    training_log: pd.DataFrame,
    allowed_templates: frozenset[str] | set[str],
    min_support_ratio: float,
    min_confidence_ratio: float,
) -> dict[str, dict[Any, dict[str, int]]]:
    return pm4py.discover_declare(
        training_log,
        allowed_templates=set(allowed_templates),
        min_support_ratio=min_support_ratio,
        min_confidence_ratio=min_confidence_ratio,
    )


def imperative_acceptance(log: pd.DataFrame, model: ImperativeModel) -> list[bool]:
    if log.empty:
        return []
    diagnostics = pm4py.conformance_diagnostics_token_based_replay(
        log,
        model.net,
        model.initial_marking,
        model.final_marking,
    )
    known_activities = {
        str(transition.label)
        for transition in getattr(model.net, "transitions", [])
        if getattr(transition, "label", None)
    }
    traces = dataframe_to_traces(log)
    results: list[bool] = []
    for trace, item in zip(traces, diagnostics, strict=True):
        if not set(trace) <= known_activities:
            results.append(False)
            continue
        if "trace_is_fit" in item:
            results.append(bool(item["trace_is_fit"]))
        elif "is_fit" in item:
            results.append(bool(item["is_fit"]))
        else:
            results.append(float(item.get("trace_fitness", 0.0)) >= 1.0)
    return results


def declarative_acceptance(
    log: pd.DataFrame,
    model: dict[str, dict[Any, dict[str, int]]],
    known_activities: set[str] | None = None,
) -> list[bool]:
    return [
        _trace_satisfies_model(trace, model, known_activities)
        for trace in dataframe_to_traces(log)
    ]


def imperative_complexity(model: ImperativeModel) -> dict[str, float]:
    places = list(getattr(model.net, "places", []))
    transitions = list(getattr(model.net, "transitions", []))
    arcs = list(getattr(model.net, "arcs", []))
    visible = [transition for transition in transitions if getattr(transition, "label", None)]
    hidden = [transition for transition in transitions if not getattr(transition, "label", None)]

    bpmn_nodes = list(_call_or_empty(model.bpmn, "get_nodes"))
    bpmn_flows = list(_call_or_empty(model.bpmn, "get_flows"))
    gateways = [node for node in bpmn_nodes if "gateway" in node.__class__.__name__.lower()]
    tasks = [node for node in bpmn_nodes if "task" in node.__class__.__name__.lower()]
    xor_gateways = [node for node in gateways if _is_gateway_kind(node, ("xor", "exclusive"))]
    and_gateways = [node for node in gateways if _is_gateway_kind(node, ("and", "parallel"))]
    or_gateways = [node for node in gateways if _is_gateway_kind(node, ("or", "inclusive"))]

    outgoing_counts = _bpmn_outgoing_counts(bpmn_nodes, bpmn_flows)
    split_gateways = [node for node in gateways if outgoing_counts.get(node, 0) > 1]
    cfc = 0
    for gateway in split_gateways:
        outgoing = outgoing_counts[gateway]
        if gateway in xor_gateways:
            cfc += outgoing
        elif gateway in or_gateways:
            cfc += (2**outgoing) - 1
        elif gateway in and_gateways:
            cfc += 1
        else:
            cfc += outgoing

    node_count = len(bpmn_nodes)
    flow_count = len(bpmn_flows)
    return {
        "petri_places": len(places),
        "petri_transitions": len(transitions),
        "petri_visible_transitions": len(visible),
        "petri_hidden_transitions": len(hidden),
        "petri_arcs": len(arcs),
        "bpmn_nodes": node_count,
        "bpmn_flows": flow_count,
        "bpmn_tasks": len(tasks),
        "bpmn_gateways": len(gateways),
        "bpmn_xor_gateways": len(xor_gateways),
        "bpmn_and_gateways": len(and_gateways),
        "bpmn_or_gateways": len(or_gateways),
        "bpmn_split_gateways": len(split_gateways),
        "control_flow_complexity": cfc,
        "bpmn_density": flow_count / (node_count * (node_count - 1)) if node_count > 1 else 0.0,
    }


def imperative_native_quality(log: pd.DataFrame, model: ImperativeModel) -> dict[str, float | None]:
    quality: dict[str, float | None] = {
        "native_fitness": None,
        "native_precision": None,
        "native_generalization": None,
        "native_simplicity": None,
    }
    try:
        fitness = pm4py.fitness_token_based_replay(
            log,
            model.net,
            model.initial_marking,
            model.final_marking,
        )
        quality["native_fitness"] = float(
            fitness.get("log_fitness", fitness.get("average_trace_fitness", 0.0))
        )
    except Exception:
        pass
    try:
        quality["native_precision"] = float(
            pm4py.precision_token_based_replay(
                log,
                model.net,
                model.initial_marking,
                model.final_marking,
            )
        )
    except Exception:
        pass
    try:
        quality["native_generalization"] = float(
            pm4py.generalization_tbr(
                log,
                model.net,
                model.initial_marking,
                model.final_marking,
            )
        )
    except Exception:
        pass
    try:
        quality["native_simplicity"] = float(
            pm4py.simplicity_petri_net(
                model.net,
                model.initial_marking,
                model.final_marking,
            )
        )
    except Exception:
        pass
    return quality


def declarative_trace_conformance(
    log: pd.DataFrame, model: dict[str, dict[Any, dict[str, int]]]
) -> dict[str, float]:
    traces = dataframe_to_traces(log)
    constraint_count = sum(len(items) for items in model.values())
    if not traces:
        return {"full_trace_conformance": 0.0, "mean_constraint_satisfaction": 0.0}
    if constraint_count == 0:
        return {"full_trace_conformance": 1.0, "mean_constraint_satisfaction": 1.0}

    fully_conformant = 0
    ratios: list[float] = []
    for trace in traces:
        satisfied = sum(
            1
            for template, parameters in _iter_model_parameters(model)
            if _constraint_holds(template, _parameter_tuple(parameters), trace)
        )
        ratios.append(satisfied / constraint_count)
        if satisfied == constraint_count:
            fully_conformant += 1
    return {
        "full_trace_conformance": fully_conformant / len(traces),
        "mean_constraint_satisfaction": sum(ratios) / len(ratios),
    }


def _trace_satisfies_model(
    trace: Trace,
    model: dict[str, dict[Any, dict[str, int]]],
    known_activities: set[str] | None = None,
) -> bool:
    if known_activities is not None and not set(trace) <= known_activities:
        return False
    for template, parameters in _iter_model_parameters(model):
        if not _constraint_holds(template, _parameter_tuple(parameters), trace):
            return False
    return True


def _iter_model_parameters(model: dict[str, dict[Any, dict[str, int]]]):
    for template, parameter_map in model.items():
        for parameters in parameter_map:
            yield template, parameters


def _constraint_holds(template: str, parameters: tuple[str, ...], trace: Trace) -> bool:
    if template == "init":
        return bool(trace) and trace[0] == parameters[0]
    if template == "exactly_one":
        return trace.count(parameters[0]) == 1
    if template == "existence":
        return trace.count(parameters[0]) >= 1
    if template == "absence":
        return trace.count(parameters[0]) == 0
    if len(parameters) < 2:
        return True

    left, right = parameters[:2]
    if template == "responded_existence":
        return left not in trace or right in trace
    if template == "response":
        left_positions = [index for index, event in enumerate(trace) if event == left]
        return all(any(event == right for event in trace[index + 1 :]) for index in left_positions)
    if template == "precedence":
        right_positions = [index for index, event in enumerate(trace) if event == right]
        return all(any(event == left for event in trace[:index]) for index in right_positions)
    if template == "succession":
        left_positions = [index for index, event in enumerate(trace) if event == left]
        right_positions = [index for index, event in enumerate(trace) if event == right]
        response_holds = all(
            any(event == right for event in trace[index + 1 :])
            for index in left_positions
        )
        precedence_holds = all(
            any(event == left for event in trace[:index])
            for index in right_positions
        )
        return response_holds and precedence_holds
    if template == "coexistence":
        return (left in trace) == (right in trace)
    if template == "noncoexistence":
        return not (left in trace and right in trace)
    if template == "nonsuccession":
        return not any(
            event == left and any(later == right for later in trace[index + 1 :])
            for index, event in enumerate(trace)
        )
    return True


def _parameter_tuple(parameters: Any) -> tuple[str, ...]:
    if isinstance(parameters, tuple):
        return tuple(str(item) for item in parameters)
    if isinstance(parameters, list):
        return tuple(str(item) for item in parameters)
    return (str(parameters),)


def _call_or_empty(obj: Any, method_name: str) -> list[Any]:
    method = getattr(obj, method_name, None)
    if not method:
        return []
    return list(method())


def _is_gateway_kind(node: Any, hints: tuple[str, ...]) -> bool:
    haystack = f"{node.__class__.__name__} {getattr(node, 'name', '')} {str(node)}".lower()
    return any(hint in haystack for hint in hints)


def _bpmn_outgoing_counts(nodes: list[Any], flows: list[Any]) -> dict[Any, int]:
    counts = {node: 0 for node in nodes}
    for flow in flows:
        source = _flow_endpoint(flow, "source")
        if source in counts:
            counts[source] += 1
    return counts


def _flow_endpoint(flow: Any, name: str) -> Any:
    getter = getattr(flow, f"get_{name}", None)
    if getter:
        return getter()
    return getattr(flow, name, None)
