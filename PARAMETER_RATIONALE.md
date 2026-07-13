# Parameter Rationale

This note documents why the main experimental parameters were chosen. The
parameters are fixed design choices, not optimized defaults.

## Why synthetic logs?

The research question requires knowing whether a rare trace is valid process
variability or accidental noise. A controlled synthetic specification provides
this ground truth. Real-life logs are important for external validation, but
they usually do not provide complete labels for all valid variants and all
recording errors. For that reason, the main experiment uses a documented
synthetic ground truth and treats real-life validation as future work.

Relevant sources:

- Augusto et al. (2019): process discovery benchmarks depend strongly on
  algorithms, logs, measures, and experimental setup.
- Sommers et al. (2025): ground-truth process data is useful when the goal is
  to assess process mining techniques under known valid behavior and known
  deviations.

## Variability levels

The main variability probabilities are:

```text
p = 0.00, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50
```

The levels have the following interpretation:

- `p=0.00`: no-variability baseline.
- `p=0.05` and `p=0.10`: rare but valid behavior.
- `p=0.20` to `p=0.50`: medium to strong variability.

`p` is a probability for one scenario-specific generation mechanism. It is not
a directly comparable amount of total variability across scenarios: optional
has two independent decisions, mixed has several decisions, and order/rework
each have one. Observed variant count and normalized entropy therefore describe
the realized log variability when scenarios are compared.

With 350 training traces, a single rare mechanism at `p=0.05` is expected to
appear in about:

```text
350 * 0.05 = 17.5 traces
```

This is intentional. The behavior is visible in the log, but still close to the
threshold region where a discovery algorithm may treat it as too infrequent.

## Training and test sizes

Each main condition uses:

```text
350 training traces
150 positive test traces
150 negative test traces
10 random seeds
```

The training size is a compromise. It is large enough that rare variants can
appear in most non-zero conditions, but small enough that incomplete-log and
threshold effects remain observable. Ten seeds reduce dependence on one random
sample while keeping all generated logs and models inspectable. Ten seeds are
used for descriptive spread and threshold-instability checks; they are not
presented as a large inferential sample or as a basis for strong significance
claims.

The two test classes contain 150 traces each so that classification measures are
not driven by an arbitrary valid/invalid class imbalance. At this size, every
valid variant and every documented mutation family is exercised repeatedly,
while all 300 decisions per condition remain practical to inspect. The positive
test set is balanced across valid variants. The negative set is balanced across
the documented mutation families, with family counts differing by at most one.
It remains a supplementary acceptance/rejection test rather than a replacement
for standard fitness and precision. Consequently, F1 is a controlled
variant-coverage score, not a performance estimate weighted by natural
frequencies.

Because the invalid families are hand-designed, the analysis also reports
mutation-macro rejection and leave-one-mutation-family-out F1. The latter omits
each family once and records the minimum, maximum, and range. A small range
shows that no single mutation family drives the reported F1; it does not prove
coverage of the complete invalid trace language.

Declare is evaluated under two explicitly separated policies. The primary
model-semantics result is open-world: a label that is not constrained by the
discovered Declare model is not rejected merely because it was absent from the
training alphabet. The supplementary closed-world result adds an alphabet rule
that rejects such labels. This rule can be useful in a deployment that treats
the observed alphabet as exhaustive, but it is not itself a discovered Declare
constraint. Reporting both policies prevents the evaluation wrapper from being
mistaken for model behavior.

## Inductive Miner threshold

The main variability experiment uses Inductive Miner threshold `0.0` because all
sampled variants are valid under the synthetic ground truth. A positive
threshold would intentionally allow the miner to ignore infrequent behavior,
which would make it harder to separate valid variability from noise.

The separate noise experiment varies the Inductive Miner threshold over:

```text
0.0, 0.1, 0.2
```

This isolates the effect of filtering accidental deviations.

Relevant source:

- Leemans et al. (2014): Inductive Miner addresses infrequent behavior through
  filtering.

## Declare support and confidence

The main Declare settings are:

```text
support = 0.05
confidence = 0.95
```

Support `0.05` is deliberately aligned with the rare-variant region. With 350
training traces, it corresponds to roughly 17 to 18 supporting cases.
Confidence `0.95` requires a relation to hold almost always while allowing
small sampling variation.

These values are not claimed to be universal defaults. They are fixed main
settings. The rework sensitivity experiment varies them:

```text
support    = 0.03, 0.05, 0.07
confidence = 0.90, 0.95, 0.99
```

The support values bracket the main value `0.05` symmetrically in the rare-
behavior region. Confidence `0.90`, `0.95`, and `0.99` represent a more
permissive, central, and stricter requirement. Rework probabilities `0.03`,
`0.05`, `0.07`, and `0.10` span the same transition region. Crossing these
values tests whether the reported change is robust rather than a numerical
coincidence between rework probability `0.05` and support `0.05`.

Relevant source:

- Maggi et al. (2012): declarative process discovery uses relevance measures
  such as support and confidence to prune candidate constraints.

## Declare template language

The main configuration uses this fixed PM4Py-compatible template set:

```text
init, exactly_one, responded_existence, response, precedence, succession,
coexistence
```

`responded_existence` is required by PM4Py to derive `coexistence`.
`succession` represents the paired response/precedence relation. Negative
templates are not included because this experiment fixes a positive
occurrence/co-occurrence/ordering language; adding them would change the
candidate space. This is a deliberately limited language, not a claim to cover
all Declare templates.

## Canonical Declare constraint count

PM4Py's discovered-model object can serialize one semantic rule through several
relation records. In particular, a symmetric coexistence relation can appear in
both directions, while composite templates can appear together with the
primitive relations from which they are derived. The experiment therefore
retains two structural counts:

- the raw relation-record count, which describes the PM4Py output exactly;
- the canonical constraint count, which is the primary complexity proxy.

Canonicalization collapses reverse duplicates of symmetric relations. When a
composite relation is present, it keeps `succession` rather than also counting
the matching `response` and `precedence`, and keeps `coexistence` rather than
also counting both directed `responded_existence` relations. The complete raw
model is still used for conformance and trace acceptance, so this choice changes
only structural counting. Both counts remain in `results.csv` to make the
measurement decision auditable.

As a representation sensitivity check, the results also include a
primitive-preferred count. It removes `succession` when the corresponding
`response` and `precedence` are present and removes `coexistence` when both
directed `responded_existence` relations are present. The two policies produce
different absolute counts but should support the same directional conclusions.

## Why not replace the synthetic experiment with a real-life log?

A real-life log would be useful as an additional external-validity check, but it
would not replace the controlled experiment. The main question is about rare
valid behavior versus noise. Without a labeled ground truth, a real-life log
would make it difficult to decide whether a rare trace is valid, exceptional, or
erroneous.

For the final seminar version, a real-life log would likely add too much scope:
case notion, preprocessing, activity abstraction, missing labels, and a second
interpretation layer. A realistic extension would be a short future-work
section or a small optional robustness appendix, not a replacement for the main
synthetic design.
