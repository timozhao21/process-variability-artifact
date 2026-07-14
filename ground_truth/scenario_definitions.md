# Ground-Truth Specification

This folder documents the synthetic valid behavior used by the experiment. The
ground truth is a generative process specification, not a process model mined by
PM4Py. The generator samples traces from the variants below and then converts
them into CSV/XES event logs with case ids, activity names, timestamps, and an
explicit event index.

The folder also contains hand-authored BPMN and Declare ground-truth
specifications:

- `bpmn/*.bpmn` documents the intended control-flow structure per scenario.
- `declare/*.declare.json` documents the intended declarative constraint
  projection per scenario.
- `visuals/*_overview.png` and `.pdf` provide reviewer-friendly visual
  companions for the BPMN specifications.

These files are explanatory ground-truth specifications. The exact finite trace
language used by the generator remains the variant table in `valid_variants.csv`.
For a compact file overview, see `ground_truth_index.csv`.

## Event-Log Structure

Each generated process instance is one case. Each row in the CSV log is one
event. The exported logs use the following columns:

- `case:concept:name`: case id, for example `train_00000`
- `concept:name`: activity name, for example `Register`
- `time:timestamp`: synthetic timestamp, monotonic within the case
- `event_index`: explicit event order within the case

Timestamps are synthetic and are used only to provide a valid event-log
structure and deterministic ordering. The experiment does not analyze temporal
performance.

## Baseline

```text
Register -> Check -> Assess -> Decide -> Notify -> Archive
```

## Optional Scenario

The optional scenario adds two optional activities:

- `Request documents` may occur after `Check`.
- `Optional review` may occur after `Assess`.

Each optional activity is included independently with probability `p`.

## Order Scenario

The order scenario swaps `Check` and `Assess` with probability `p`.

## Rework Scenario

The rework scenario adds a repeated assessment path with probability `p`:

```text
Register -> Check -> Assess -> Manual review -> Rework -> Assess -> Decide -> Notify -> Archive
```

## Mixed Scenario

The mixed scenario is a supplementary robustness setting. It combines the three
valid variability mechanisms instead of isolating them:

- `Check` and the first `Assess` may be swapped with probability `p`.
- `Request documents` may occur after `Check` with probability `p`.
- A rework path may occur after the first `Assess` with probability `p`.
- If the rework path is not taken, a standalone `Optional review` may occur after
  the first `Assess` with probability `p`.

This creates twelve valid variants when `p > 0`. The exact finite trace language
is listed in `valid_variants.csv`.

For binary indicators `o` (swapped order), `d` (optional documents), and `m`
(optional review), a non-rework variant has unconditional probability
`p**(o+d+m) * (1-p)**(4-o-d-m)`. If rework is present, optional review is not
sampled and the variant probability is
`p**(o+d+1) * (1-p)**(2-o-d)`. The eight non-rework variants sum to `1-p`; the
four rework variants sum to `p`. These expressions follow directly from the
independent Bernoulli sampling rules: the exponent of `p` counts present
mechanisms and the exponent of `1-p` counts absent decisions, including `r=0`
for a non-rework variant.

## Noise Scenario

The noise scenario is separate from valid variability. It starts from the
baseline trace and applies accidental log corruptions, such as deleting an
event, inserting an extra event, or swapping adjacent events. Noise traces are
not treated as valid process behavior.
