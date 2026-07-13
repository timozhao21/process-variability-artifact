import pandas as pd

from spaghetti_experiment.analysis import analyze_results


def test_analysis_writes_baseline_outputs_without_complexity_index(tmp_path) -> None:
    input_path = tmp_path / "results.csv"
    output_dir = tmp_path / "analysis"
    pd.DataFrame(
        [
            {
                "scenario": "order",
                "variability": 0.0,
                "seed": 1,
                "inductive_noise_threshold": 0.0,
                "declare_min_support_ratio": 0.05,
                "declare_min_confidence_ratio": 0.95,
                "imperative_bpmn_nodes": 8,
            },
            {
                "scenario": "order",
                "variability": 0.1,
                "seed": 1,
                "inductive_noise_threshold": 0.0,
                "declare_min_support_ratio": 0.05,
                "declare_min_confidence_ratio": 0.95,
                "imperative_bpmn_nodes": 10,
            },
        ]
    ).to_csv(input_path, index=False)

    analyze_results(input_path, output_dir)

    analyzed = pd.read_csv(output_dir / "results_with_baselines.csv")
    assert "imperative_bpmn_nodes_change_from_p0" in analyzed
    seed_summary = pd.read_csv(output_dir / "key_metric_seed_summary.csv")
    assert set(seed_summary["metric"]) == {"imperative_bpmn_nodes"}
    assert seed_summary.loc[0, "n_seeds"] == 1
    assert seed_summary.loc[0, "std"] == 0.0
    assert not (output_dir / "results_with_indices.csv").exists()
    assert not (output_dir / "crossovers.csv").exists()
