"""Check that every exported result can be traced to its referenced artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


KEY_COLUMNS = [
    "scenario",
    "variability",
    "seed",
    "inductive_noise_threshold",
    "declare_min_support_ratio",
    "declare_min_confidence_ratio",
]

REQUIRED_REVISED_METRICS = [
    "declarative_declare_relation_records_raw",
    "declarative_declare_constraints",
    "declarative_declare_constraints_primitive_preferred",
    "declarative_test_f1",
    "declarative_test_f1_leave_one_family_out_range",
    "declarative_closed_world_test_f1",
]


def verify_rework_threshold_evidence(
    results: pd.DataFrame, experiment_dir: Path, artifact_index: pd.DataFrame
) -> list[str]:
    """Guard Figure 4 counts and its constraint-specific confidence explanation."""
    errors: list[str] = []
    rework = results[results["scenario"] == "rework"]
    low = rework[rework["variability"].round(8) == 0.05]
    medium = rework[rework["variability"].round(8) == 0.10]
    primary = "declarative_declare_constraints"
    primitive = "declarative_declare_constraints_primitive_preferred"
    if len(low) != 10 or len(medium) != 10:
        return ["main rework threshold rows do not contain ten seeds at p=0.05 and p=0.10"]
    low_primary = low[primary].value_counts().to_dict()
    low_primitive = low[primitive].value_counts().to_dict()
    if low_primary != {37: 7, 64: 3}:
        errors.append(f"unexpected Figure 4 primary counts at p=0.05: {low_primary}")
    if low_primitive != {67: 7, 96: 3}:
        errors.append(f"unexpected Figure 4 primitive counts at p=0.05: {low_primitive}")
    if set(medium[primary]) != {64} or set(medium[primitive]) != {96}:
        errors.append("unexpected Figure 4 counts at rework p=0.10")
    for _, result_row in low.iterrows():
        matches = artifact_index[
            (artifact_index[KEY_COLUMNS] == result_row[KEY_COLUMNS].to_dict()).all(axis=1)
        ]
        if len(matches) != 1:
            errors.append(f"missing artifact index row for rework seed {result_row['seed']}")
            continue
        model_path = experiment_dir / str(matches.iloc[0]["declare_model_json"])
        training_path = experiment_dir / str(matches.iloc[0]["training_log_csv"])
        try:
            model = json.loads(model_path.read_text(encoding="utf-8"))
            training = pd.read_csv(training_path)
            counts = training.groupby("case:concept:name")["concept:name"].apply(
                lambda values: int((values == "Assess").sum())
            )
            support_count = int((counts > 0).sum())
            confidence_count = int((counts == 1).sum())
            retained = any(
                item.get("template") == "exactly_one"
                and item.get("parameters") == ["Assess"]
                for item in model
            )
            if support_count != 350:
                errors.append(
                    f"unexpected exactly_one(Assess) support for seed {result_row['seed']}: {support_count}"
                )
            expected_retained = confidence_count / support_count >= 0.95
            if retained != expected_retained:
                errors.append(
                    f"exactly_one(Assess) retention does not match confidence for seed {result_row['seed']}"
                )
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"could not inspect rework constraint evidence for seed {result_row['seed']}: {exc}")
    return errors


def verify_experiment(experiment_dir: Path) -> list[str]:
    errors: list[str] = []
    results_path = experiment_dir / "results.csv"
    index_path = experiment_dir / "artifact_index.csv"
    analysis_dir = experiment_dir / "analysis"

    for path in (results_path, index_path, experiment_dir / "manifest.json"):
        if not path.is_file():
            errors.append(f"missing required file: {path}")
    if errors:
        return errors

    results = pd.read_csv(results_path)
    index = pd.read_csv(index_path)
    for column in KEY_COLUMNS:
        if column not in results or column not in index:
            errors.append(f"missing traceability column: {column}")
    for column in REQUIRED_REVISED_METRICS:
        if column not in results:
            errors.append(f"missing revised result metric: {column}")
    if errors:
        return errors

    if (
        results["declarative_declare_constraints"]
        > results["declarative_declare_relation_records_raw"]
    ).any():
        errors.append("canonical Declare count exceeds raw relation-record count")

    result_keys = set(map(tuple, results[KEY_COLUMNS].itertuples(index=False, name=None)))
    index_keys = set(map(tuple, index[KEY_COLUMNS].itertuples(index=False, name=None)))
    if result_keys != index_keys:
        errors.append("result rows and artifact-index rows do not have identical condition keys")
    result_lookup = results.set_index(KEY_COLUMNS)

    artifact_columns = [
        column
        for column in index.columns
        if column.endswith(("_csv", "_xes", "_bpmn", "_pnml", "_json"))
    ]
    for row_number, row in index.iterrows():
        for column in artifact_columns:
            relative_path = Path(str(row[column]))
            if relative_path.is_absolute() or ".." in relative_path.parts:
                errors.append(f"unsafe artifact path at row {row_number}: {relative_path}")
            elif not (experiment_dir / relative_path).is_file():
                errors.append(f"missing artifact at row {row_number}: {relative_path}")
        diagnostics_path = experiment_dir / Path(str(row["behavioral_diagnostics_csv"]))
        if diagnostics_path.is_file():
            condition_key = tuple(row[column] for column in KEY_COLUMNS)
            expected = result_lookup.loc[condition_key]
            diagnostics = pd.read_csv(diagnostics_path)
            required_diagnostic_columns = {
                "case_type",
                "mutation_family",
                "imperative_accepts",
                "declare_model_accepts",
                "declare_closed_world_accepts",
            }
            if not required_diagnostic_columns <= set(diagnostics.columns):
                errors.append(f"invalid behavioral diagnostics schema at row {row_number}")
                continue
            counts = diagnostics["case_type"].value_counts()
            if counts.get("positive", 0) != int(expected["positive_test_traces"]):
                errors.append(f"wrong positive diagnostic count at row {row_number}")
            if counts.get("negative", 0) != int(expected["negative_test_traces"]):
                errors.append(f"wrong negative diagnostic count at row {row_number}")
            family_counts = diagnostics.loc[
                diagnostics["case_type"] == "negative", "mutation_family"
            ].value_counts()
            if not family_counts.empty and family_counts.max() - family_counts.min() > 1:
                errors.append(f"unbalanced mutation families at row {row_number}")

    summary_path = analysis_dir / "summary.csv"
    seed_summary_path = analysis_dir / "key_metric_seed_summary.csv"
    if not summary_path.is_file():
        errors.append(f"missing analysis summary: {summary_path}")
    if not seed_summary_path.is_file():
        errors.append(f"missing key seed summary: {seed_summary_path}")
    if experiment_dir.name == "main":
        errors.extend(verify_rework_threshold_evidence(results, experiment_dir, index))
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify exported experiment artifacts.")
    parser.add_argument("--outputs", default="outputs", help="Directory containing experiment outputs.")
    args = parser.parse_args()

    output_root = Path(args.outputs)
    experiments = sorted(path for path in output_root.iterdir() if path.is_dir())
    if not experiments:
        raise SystemExit(f"No experiment directories found under {output_root}")

    failures = 0
    for experiment_dir in experiments:
        errors = verify_experiment(experiment_dir)
        if errors:
            failures += 1
            print(f"FAILED {experiment_dir.name}")
            for error in errors:
                print(f"  - {error}")
        else:
            row_count = len(pd.read_csv(experiment_dir / "results.csv"))
            print(f"OK {experiment_dir.name}: {row_count} result rows are traceable")
    if failures:
        raise SystemExit(f"Artifact verification failed for {failures} experiment(s).")


if __name__ == "__main__":
    main()
