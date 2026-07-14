# Spaghetti Experiment

This repository contains the complete experimental artifacts for a reproducible
comparison of automatically discovered imperative and declarative process
models under different types and prevalence levels of valid variants.

Research question:

> How do the type and prevalence of valid process variants affect structural
> proxies and controlled behavioral discrimination in automatically discovered
> imperative and declarative process models?

The experiment studies structural proxies and controlled trace-level behavioral
discrimination. It does not include a user study; therefore, model
understandability is only addressed through structural proxy metrics.

The parameter choices are documented in `PARAMETER_RATIONALE.md`.

## Reproducibility record

The paper-facing artifact snapshot is identified as
`paper-revision-2026-07-13`. The submitted result manifests record the Python,
PM4Py and pandas versions used for each run. When this folder is published to a
remote repository, the release or commit identifier should be added to the
paper's artifact citation; the local snapshot remains runnable without a remote
service.

The intended public repository for the artifact is:
<https://github.com/timozhao21/process-variability-artifact>

## Quick Start

The archive already contains all logs, discovered models, per-trace decisions,
per-seed results, and analyses reported in the paper. A reviewer can therefore
verify the submitted results without first rerunning the experiment. Run all
commands below from the repository root:

```bash
cd <repository-root>
```

### 1. Install the pinned environment

Python 3.12 or newer is required. The submitted runs used Python 3.12.10 and
PM4Py 2.7.22.4; every run records its exact environment in `manifest.json`.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -e . --no-deps
```

### 2. Verify the included artifact

The first command checks the implementation. The second checks that every
reported condition has its expected logs, models, test decisions, trace counts,
and balanced mutation families.

```bash
.venv/bin/python -m pytest -q
.venv/bin/python verify_artifacts.py --outputs outputs
```

Expected result: all tests pass and the verifier prints one `OK` line for each
included run (`main`, `mixed`, `noise`, `pilot`, `rework_sensitivity`,
`training_size_100`, `training_size_350`, and `training_size_1000`) without an
error or traceback.

For the main run, verification also checks the rework-threshold evidence shown
in the paper:
at rework $p=0.05$, seven seeds have 37/67 composite-/primitive-preferred
constraints and three have 64/96; at $p=0.10$, all ten seeds have 64/96.
It additionally inspects each referenced training log and Declare JSON to verify
that the reported `exactly_one(Assess)` support is 350 and that retention agrees
with the observed confidence ratio at the configured 0.95 threshold.

### 3. Run a small inspectable example

The pilot has 27 conditions and completes in seconds on the development
machine. It is the fastest way to exercise generation, discovery, evaluation,
and artifact export end to end.

```bash
.venv/bin/python run_experiment.py \
  --config configs/pilot.json \
  --output outputs

.venv/bin/python analyze_results.py \
  --input outputs/pilot/results.csv \
  --output outputs/pilot/analysis
```

Afterwards, inspect `outputs/pilot/results.csv`, select one matching row in
`outputs/pilot/artifact_index.csv`, and open its folder under
`outputs/pilot/artifacts/`.

### 4. Reproduce all reported experiments

```bash
for config in main mixed noise rework_sensitivity \
  training_size_100 training_size_350 training_size_1000; do
  .venv/bin/python run_experiment.py \
    --config "configs/${config}.json" \
    --output outputs
done

for experiment in main mixed noise rework_sensitivity \
  training_size_100 training_size_350 training_size_1000; do
  .venv/bin/python analyze_results.py \
    --input "outputs/${experiment}/results.csv" \
    --output "outputs/${experiment}/analysis"
done

.venv/bin/python summarize_training_size_sensitivity.py
.venv/bin/python generate_paper_figures.py
.venv/bin/python verify_artifacts.py --outputs outputs
```

The summed recorded condition runtime, including the three training-size runs,
is about 4.1 minutes on the development machine; file export and hardware
differences can increase wall-clock time, so allowing several minutes is
appropriate. The focused rework-sensitivity run is the largest part with 450
conditions. Figure generation writes the paper plots to `../paper/figures/`
and the ground-truth companions to `ground_truth/visuals/`.

The paper-facing figures include `structural_summary.pdf`,
`rework_threshold_effect.pdf`, `representative_discovered_models.pdf` and
`rework_sensitivity_heatmap.pdf`. The representative-model figure is generated
from exported BPMN and Declare discovery files in `outputs/main/artifacts/`; it
is not a hand-drawn ground-truth model. The complete models, CSV/XES logs and
per-seed decisions remain in the corresponding artifact folders.

The broader `configs/sensitivity.json` file is a prepared extension and is not
part of the reported result set. The reported focused sensitivity experiment is
`configs/rework_sensitivity.json` and its complete output is included.

Figure 1 in the paper uses the manually prepared image at ../paper/figures/image.pdf;
it is intentionally not regenerated by the figure script. The other paper plots
remain reproducible from the commands above.

## Repository Structure

```text
configs/
  pilot.json          small pilot run with exported artifacts
  main.json           main experiment
  noise.json          separate observation-noise experiment
  mixed.json          supplementary mixed-variability experiment
  rework_sensitivity.json reported focused threshold-robustness experiment
  training_size_100.json focused rework run with 100 training traces
  training_size_350.json focused rework run with 350 training traces
  training_size_1000.json focused rework run with 1,000 training traces
  sensitivity.json    prepared sensitivity configuration
ground_truth/          documented synthetic valid-behavior specification
  bpmn/                hand-authored BPMN ground-truth files
  declare/             hand-authored Declare ground-truth files
  visuals/             generated visual companions for the ground truth
PARAMETER_RATIONALE.md rationale for p-levels, thresholds, and trace counts
src/spaghetti_experiment/
  config.py           configuration loading and validation
  generator.py        synthetic process and event-log generation
  models.py           imperative and declarative model discovery
  metrics.py          complexity, canonicalization and quality metrics
  runner.py           experiment execution
  analysis.py         aggregation and plot generation
  artifacts.py        CSV, XES, PNML, BPMN, and JSON export
generate_paper_figures.py regenerates paper and ground-truth figures
summarize_training_size_sensitivity.py combines the three training-size runs
tests/                 unit and integration tests
outputs/               generated experiment outputs
```

The Declare structural metrics are computed from the observed activity alphabet
and the configured template language. Density uses the number of possible unary,
directed-pair and unordered-pair constraints as its denominator. The primary
count is composite-preferred; the primitive-preferred count is a representation
sensitivity check. Graph degree treats each binary relation as an undirected
activity connection, while relation overlap counts additional canonical binary
relations on the same endpoint pair. The exact implementation is in
`src/spaghetti_experiment/metrics.py`, with unit tests in `tests/test_metrics.py`.

## Experiment Design

The baseline process is:

```text
Register -> Check -> Assess -> Decide -> Notify -> Archive
```

The experiment varies this process through three isolated valid variability
scenarios and one supplementary mixed scenario:

- `optional`: `Request documents` and `Optional review` may occur as optional
  activities.
- `order`: `Check` and `Assess` may switch order.
- `rework`: after `Assess`, a rework path may add `Manual review`, `Rework`,
  and a second `Assess`.
- `mixed`: optional activities, alternative ordering, and one-time rework are
  combined in a supplementary robustness scenario.

A separate `noise` scenario introduces observation errors into the training log.
Noise is not treated as valid process behavior.

The mixed generator uses independent Bernoulli(`p`) decisions for swapped order,
optional documents and rework. When rework is absent, a fourth Bernoulli(`p`)
decision controls optional review; when rework is present, the distinct manual
review is part of the rework path. This yields eight non-rework variants and four rework
variants. For binary indicators `o`, `d`, and `m`, an unconditional non-rework
variant has probability `p**(o+d+m) * (1-p)**(4-o-d-m)`. A rework variant has
probability `p**(o+d+1) * (1-p)**(2-o-d)`; optional review is not sampled when
rework is present. The non-rework and rework groups therefore sum to `1-p` and
`p`, respectively. These expressions are derived directly from the generator's
independent Bernoulli sampling rules: the exponent of `p` counts present
mechanisms, while the exponent of `1-p` counts absent decisions, including
`r=0` in the non-rework case.

For the noise scenario, a corrupted case receives exactly one uniformly selected
operator: deletion at a uniform event position, an adjacent swap at a uniform
pair position, or insertion at a uniform position with an activity sampled from
`Request documents`, `Optional review`, `Manual review`, `Rework` and `Escalate`.

For each condition, the generator creates:

- a training log,
- a balanced positive test set with valid traces,
- a negative test set balanced across documented invalid near-miss families.

Seeds are derived with SHA-256 so that training, positive test, and negative
test generation use separate random streams while remaining reproducible.

`p` is a probability of a scenario-specific generation mechanism, not a common
scale of total log variability across scenarios. The optional and mixed
scenarios contain multiple random decisions per case, whereas order and rework
contain one. Use observed variant count and entropy when comparing realized log
variability across scenarios.

## Event-Log Structure

The synthetic generator first samples traces from the documented valid-behavior
specification in `ground_truth/`. Before discovery, each trace collection is
converted into a PM4Py-compatible event-log table. Each generated process
instance is one case, and each row is one event.

The CSV logs use these columns:

- `case:concept:name`: case id, for example `train_00000`
- `concept:name`: activity name, for example `Register`
- `time:timestamp`: synthetic timestamp, monotonic within each case
- `event_index`: explicit event order inside the case

Timestamps are synthetic and are used only to provide a valid event-log
structure and deterministic ordering. The experiment does not analyze temporal
performance.

## Configurations

`configs/pilot.json`

- 3 scenarios: `optional`, `order`, `rework`
- 3 prevalence levels: `0.0`, `0.2`, `0.5`
- 3 seeds
- exports per-condition logs and models

`configs/main.json`

- 3 scenarios: `optional`, `order`, `rework`
- 7 prevalence levels from `0.0` to `0.5`
- 10 seeds
- 350 training traces per condition
- 150 positive and 150 negative test traces per condition
- exports per-condition logs and models

`configs/noise.json`

- separate `noise` scenario
- variability levels `0.0`, `0.05`, `0.1`, `0.2`
- 5 seeds
- Inductive Miner noise thresholds `0.0`, `0.1`, `0.2`
- exports per-condition logs and models

`configs/sensitivity.json`

- prepared parameter-sensitivity configuration
- varies Inductive Miner threshold, Declare support, and Declare confidence
- broader all-scenario extension; it is not part of the reported final results

`configs/rework_sensitivity.json`

- focused robustness check for the rework threshold effect
- rework probabilities `0.0`, `0.03`, `0.05`, `0.07`, `0.1`
- Declare support thresholds `0.03`, `0.05`, `0.07`
- Declare confidence thresholds `0.90`, `0.95`, `0.99`
- 10 seeds per condition
- exports per-condition logs and models

`configs/mixed.json`

- supplementary scenario combining optional activities, alternative ordering,
  and one-time rework
- same prevalence levels and seeds as the main experiment
- exports per-condition logs and models

## Discovery Methods

Imperative models are discovered with PM4Py's Inductive Miner:

```python
pm4py.discover_process_tree_inductive(
    training_log,
    noise_threshold=noise_threshold,
)
```

The discovered process tree is converted to both Petri net and BPMN form.

Declarative models are discovered with PM4Py Declare discovery:

```python
pm4py.discover_declare(
    training_log,
    allowed_templates=templates,
    min_support_ratio=0.05,
    min_confidence_ratio=0.95,
)
```

PM4Py retains a candidate when its support ratio and its confidence ratio both
meet the configured minima. Support counts activated cases; confidence counts
satisfied activated cases divided by support. The exported `declare_model.json`
stores both counts for every retained constraint. The evaluator then treats the
retained constraints as a conjunction: a trace is accepted only when every
constraint is satisfied. This is a deterministic hard-decision wrapper around
the probabilistic discovery statistics, not a second discovery algorithm.

The main configuration uses these Declare templates:

- `init`
- `exactly_one`
- `responded_existence`
- `response`
- `precedence`
- `succession`
- `coexistence`

`responded_existence` is required by PM4Py when deriving `coexistence`.
`succession` captures paired response/precedence relations. The negative
templates are intentionally outside this documented positive-ground-truth
template language.

## Metrics

Both model types are evaluated on the same positive and negative test traces.
The shared behavioral metrics include:

- accuracy,
- precision,
- recall,
- F1,
- false acceptance rate,
- false rejection rate.

For Declare, the primary classification result follows the discovered model's
open-world semantics: an activity that is not mentioned by a discovered
constraint is not automatically forbidden. A second, explicitly named
closed-world result rejects labels that were absent from the training alphabet.
Reporting both policies separates model behavior from this additional deployment
rule. The rework threshold result is unchanged because the rework labels are
already present in the affected training logs.

The negative set gives each documented mutation family equal weight up to one
trace when 150 cases are not exactly divisible by the number of families. In
addition to ordinary F1, the output reports a macro rejection rate and the range
of F1 values obtained when each mutation family is omitted once. This
leave-one-family-out range measures dependence on any single hand-designed
negative family without presenting it as a standard process-mining metric.

Imperative complexity metrics include:

- BPMN nodes and flows,
- BPMN gateways,
- Petri net places, transitions, and arcs,
- control-flow complexity,
- BPMN density.
- BPMN non-task structural overhead,
- BPMN nodes per observed activity.

Declarative complexity metrics include:

- raw PM4Py relation-record count,
- canonical constraint count,
- number of template types,
- constraint density,
- constraints per observed activity,
- constraint-graph degree,
- relation overlap,
- unconstrained activities.

The raw count describes PM4Py's serialized output. The canonical count is the
primary structural measure because serialization can contain two records for
one symmetric relation and both primitive and composite forms of the same rule.
Canonicalization treats `coexistence(A,B)` and `coexistence(B,A)` as one
constraint, keeps `succession(A,B)` instead of also counting its paired
`response` and `precedence`, and keeps `coexistence` instead of also counting
both directed `responded_existence` records. It changes structural measurement
only; trace acceptance is still evaluated against the complete discovered
model.

An alternative primitive-preferred count makes the opposite representation
choice: it retains matching `response`, `precedence`, and directed
`responded_existence` relations instead of their composite forms. Both counts
are exported. The paper's directional conclusions are required to agree under
both policies; absolute counts are not interpreted across policies.

The analysis computes baseline changes from `p=0` for selected structural
metrics. It does not compute a min-max normalized complexity index. Raw values,
activity-adjusted values, and within-scenario baseline changes are the primary
evidence.

## Output Files

Each executed experiment produces:

```text
outputs/<experiment>/manifest.json
outputs/<experiment>/results.csv
outputs/<experiment>/artifact_index.csv      # when artifact export is enabled
outputs/<experiment>/analysis/summary.csv
outputs/<experiment>/analysis/results_with_baselines.csv
outputs/<experiment>/analysis/key_metric_seed_summary.csv
outputs/<experiment>/analysis/*.png
```

The included generated outputs are:

- `outputs/pilot`
- `outputs/main` with full per-condition artifacts
- `outputs/noise` with full per-condition artifacts
- `outputs/rework_sensitivity` with full per-condition artifacts
- `outputs/mixed` with full per-condition artifacts
- `outputs/training_size_100`, `outputs/training_size_350`, and
  `outputs/training_size_1000` with focused rework artifacts

The main run has 210 result rows, mixed has 70, noise has 60, the focused
rework-sensitivity run has 450, each training-size run has 50, and the pilot has
27. These counts provide a quick completeness check before inspecting individual
conditions. The combined training-size report is
`outputs/training_size_sensitivity_summary.csv`.

When artifact export is enabled, each condition folder additionally contains:

```text
training_log.csv / .xes
positive_test_log.csv / .xes
negative_test_log.csv / .xes
invalid_mutations.csv
behavioral_diagnostics.csv
imperative_model.pnml
imperative_model.bpmn
declare_model.json
```

`manifest.json` records the configuration and software versions used for the
run.

The main, noise, rework-sensitivity, mixed, and training-size runs export the
complete traceable artifact chain. Each result row can therefore be traced back
to the generated logs, discovered models, and invalid mutations used for that
condition.

## Traceability

The `ground_truth/ground_truth_index.csv` file lists the documented ground-truth
variant table, generation probabilities, BPMN files, and Declare files.

`results.csv` contains one row per condition and seed. `summary.csv` aggregates
these rows over seeds using medians and descriptive 95% percentile-bootstrap
intervals.
`key_metric_seed_summary.csv` is a compact report for the primary metrics; it
includes the number of seeds, median, mean, sample standard deviation, range,
and bootstrap interval. The intervals are descriptive because each condition
uses ten seeds.
When artifacts are exported, `artifact_index.csv` maps each condition to its
generated logs and discovered models.

`declarative_test_*` columns report model-only, open-world Declare semantics.
`declarative_closed_world_test_*` columns report the additional alphabet policy
that rejects a trace containing a label not observed in training. This naming is
intentional: the alphabet policy is an evaluation wrapper, not a discovered
Declare constraint.

`behavioral_diagnostics.csv` records every positive and negative trace together
with its mutation family and the three model decisions. It is the audit trail
for mutation-macro and leave-one-family-out results.

To inspect one condition:

1. Choose a row in `outputs/<experiment>/results.csv`.
2. Match `scenario`, `variability`, `seed`, `inductive_noise_threshold`,
   `declare_min_support_ratio`, and `declare_min_confidence_ratio` in
   `outputs/<experiment>/artifact_index.csv`.
3. Open the referenced condition folder under `outputs/<experiment>/artifacts/`.
4. Inspect the CSV/XES logs, `imperative_model.bpmn`,
   `imperative_model.pnml`, `declare_model.json`, and
   `invalid_mutations.csv`.

## Limitations

- The ground-truth process is synthetic and relatively small.
- Only control-flow behavior is studied.
- No user study on actual model understandability is included.
- Negative test traces are constructed near-miss cases.
- F1 is a balanced valid-variant coverage test, not distributional test error.
- Open-world and closed-world Declare results answer different operational
  questions and are therefore reported separately.
- Declare is evaluated with a limited template set.
- A focused rework-threshold sensitivity output with support and confidence
  variation is included. The broader all-scenario sensitivity configuration
  remains a future robustness extension.
- BPMN nodes and Declare constraints are different units; activity-adjusted
  metrics contextualize but do not make them directly comparable.

## Submission Hygiene

The submitted artifact should not include local development folders such as
`.venv/`, `.idea/`, `.pytest_cache/`, `__pycache__/`, `*.egg-info/`, or
`.Rhistory`. These files are ignored by `.gitignore` and are excluded from the
clean submission archive.
