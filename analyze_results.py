from __future__ import annotations

import argparse
from pathlib import Path

from spaghetti_experiment.analysis import analyze_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze experiment results.")
    parser.add_argument(
        "--input",
        default="outputs/pilot/results.csv",
        help="CSV file created by run_experiment.py.",
    )
    parser.add_argument(
        "--output",
        default="outputs/pilot/analysis",
        help="Directory for summaries and figures.",
    )
    args = parser.parse_args()

    analyze_results(Path(args.input), Path(args.output))
    print(f"Analysis written to {args.output}")


if __name__ == "__main__":
    main()

