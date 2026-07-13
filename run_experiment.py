from __future__ import annotations

import argparse
from pathlib import Path

from spaghetti_experiment.config import load_config
from spaghetti_experiment.runner import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the imperative-versus-declarative variability experiment."
    )
    parser.add_argument(
        "--config",
        default="configs/pilot.json",
        help="Path to a JSON experiment configuration.",
    )
    parser.add_argument(
        "--output",
        default="outputs",
        help="Root directory for generated results.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"Running {config.name!r}: {config.condition_count} conditions")
    result_path = run_experiment(config, Path(args.output))
    print(f"Results written to {result_path}")


if __name__ == "__main__":
    main()

