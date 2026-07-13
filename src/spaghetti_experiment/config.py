from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


DECLARE_TEMPLATE_DEPENDENCIES = {
    "coexistence": frozenset({"responded_existence"}),
    "noncoexistence": frozenset({"responded_existence", "coexistence"}),
    "nonsuccession": frozenset({"response", "precedence", "succession"}),
}


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    scenarios: tuple[str, ...]
    variability_levels: tuple[float, ...]
    seeds: tuple[int, ...]
    train_traces: int
    positive_test_traces: int
    negative_test_traces: int
    inductive_noise_thresholds: tuple[float, ...]
    declare_min_support_ratios: tuple[float, ...]
    declare_min_confidence_ratios: tuple[float, ...]
    declare_templates: frozenset[str]
    compute_native_quality: bool
    export_artifacts: bool
    show_progress_bars: bool

    @property
    def condition_count(self) -> int:
        return (
            len(self.scenarios)
            * len(self.variability_levels)
            * len(self.seeds)
            * len(self.inductive_noise_thresholds)
            * len(self.declare_min_support_ratios)
            * len(self.declare_min_confidence_ratios)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "scenarios": list(self.scenarios),
            "variability_levels": list(self.variability_levels),
            "seeds": list(self.seeds),
            "train_traces": self.train_traces,
            "positive_test_traces": self.positive_test_traces,
            "negative_test_traces": self.negative_test_traces,
            "inductive_noise_thresholds": list(self.inductive_noise_thresholds),
            "declare_min_support_ratios": list(self.declare_min_support_ratios),
            "declare_min_confidence_ratios": list(self.declare_min_confidence_ratios),
            "declare_templates": sorted(self.declare_templates),
            "compute_native_quality": self.compute_native_quality,
            "export_artifacts": self.export_artifacts,
            "show_progress_bars": self.show_progress_bars,
        }


def load_config(path: str | Path) -> ExperimentConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    config = ExperimentConfig(
        name=str(raw["name"]),
        scenarios=tuple(str(item) for item in raw["scenarios"]),
        variability_levels=tuple(float(item) for item in raw["variability_levels"]),
        seeds=tuple(int(item) for item in raw["seeds"]),
        train_traces=int(raw["train_traces"]),
        positive_test_traces=int(raw["positive_test_traces"]),
        negative_test_traces=int(raw["negative_test_traces"]),
        inductive_noise_thresholds=tuple(
            float(item) for item in raw["inductive_noise_thresholds"]
        ),
        declare_min_support_ratios=tuple(
            float(item) for item in raw["declare_min_support_ratios"]
        ),
        declare_min_confidence_ratios=tuple(
            float(item) for item in raw["declare_min_confidence_ratios"]
        ),
        declare_templates=frozenset(str(item) for item in raw["declare_templates"]),
        compute_native_quality=bool(raw.get("compute_native_quality", False)),
        export_artifacts=bool(raw.get("export_artifacts", False)),
        show_progress_bars=bool(raw.get("show_progress_bars", False)),
    )
    _validate_config(config)
    return config


def _validate_config(config: ExperimentConfig) -> None:
    if not config.name:
        raise ValueError("Config name must not be empty.")
    if not config.scenarios:
        raise ValueError("At least one scenario is required.")
    if not config.variability_levels:
        raise ValueError("At least one variability level is required.")
    if not config.seeds:
        raise ValueError("At least one seed is required.")
    if config.train_traces <= 0:
        raise ValueError("train_traces must be positive.")
    if config.positive_test_traces <= 0:
        raise ValueError("positive_test_traces must be positive.")
    if config.negative_test_traces <= 0:
        raise ValueError("negative_test_traces must be positive.")
    for value in config.variability_levels:
        if value < 0 or value > 1:
            raise ValueError("variability levels must be between 0 and 1.")
    for value in config.inductive_noise_thresholds:
        if value < 0 or value > 1:
            raise ValueError("inductive noise thresholds must be between 0 and 1.")
    for value in config.declare_min_support_ratios:
        if value < 0 or value > 1:
            raise ValueError("Declare support thresholds must be between 0 and 1.")
    for value in config.declare_min_confidence_ratios:
        if value < 0 or value > 1:
            raise ValueError("Declare confidence thresholds must be between 0 and 1.")
    for template, dependencies in DECLARE_TEMPLATE_DEPENDENCIES.items():
        if template not in config.declare_templates:
            continue
        missing = dependencies - config.declare_templates
        if missing:
            missing_list = ", ".join(sorted(missing))
            raise ValueError(
                f"Declare template {template!r} requires: {missing_list}."
            )
