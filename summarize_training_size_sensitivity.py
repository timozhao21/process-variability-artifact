"""Combine the focused rework training-size sensitivity runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_SIZES = (100, 350, 1000)


def summarize(output_root: Path, sizes: tuple[int, ...]) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for training_size in sizes:
        path = output_root / f"training_size_{training_size}" / "results.csv"
        frame = pd.read_csv(path)
        observed_sizes = set(frame["train_traces"].astype(int))
        if observed_sizes != {training_size}:
            raise ValueError(
                f"Expected training size {training_size} in {path}, found {observed_sizes}."
            )
        for variability, group in frame.groupby("variability", sort=True):
            perfect = group["declarative_test_f1"].eq(1.0)
            rows.append(
                {
                    "train_traces": training_size,
                    "variability": float(variability),
                    "n_seeds": int(len(group)),
                    "declarative_f1_median": float(group["declarative_test_f1"].median()),
                    "perfect_seed_count": int(perfect.sum()),
                    "perfect_seed_share": float(perfect.mean()),
                    "declare_constraints_median": float(
                        group["declarative_declare_constraints"].median()
                    ),
                    "declare_constraints_min": int(
                        group["declarative_declare_constraints"].min()
                    ),
                    "declare_constraints_max": int(
                        group["declarative_declare_constraints"].max()
                    ),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize the rework training-size sensitivity runs."
    )
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/training_size_sensitivity_summary.csv"),
    )
    parser.add_argument("--sizes", type=int, nargs="+", default=list(DEFAULT_SIZES))
    args = parser.parse_args()

    summary = summarize(args.outputs, tuple(args.sizes))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)
    print(f"Summary written to {args.output}")


if __name__ == "__main__":
    main()
