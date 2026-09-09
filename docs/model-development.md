# Model development and selection

## Objective

Train a compact classifier that detects explicit prosocial framing in professional self-descriptions and can score the full Snowflake population efficiently.

The model is intentionally narrower than a measure of actual social contribution. Its target is the language defined in `label-definition.md`.

## Why the target definition became stricter

Early pairwise comparisons answered which of two descriptions appeared more societally oriented. A pairwise winner, however, is not necessarily positive under an absolute binary definition, and a tie is not necessarily negative. This created avoidable target noise.

The final definition therefore requires explicit language about beneficiaries, community, mission, or public welfare. It prohibits inference from occupation, industry, employer, or product. This change reduced false positives caused by socially useful contexts that lacked explicit prosocial framing.

## Dataset construction

- Exact duplicate summaries are removed before splitting.
- Fixed validation and test summaries are preserved across model iterations so comparisons remain meaningful.
- Evaluation labels use four independent judgments under the same strict definition.
- Unanimous and 3–1 outcomes are accepted directly.
- Split decisions are isolated in a review file and resolved before final evaluation.
- Profile text, identifiers, and review datasets are governed inputs and are not committed to this repository.

The retained preparation scripts are in `pipeline/`.

## Model progression

### Lightweight baselines

TF–IDF/logistic regression and DistilBERT established that the signal was learnable, but both left substantial headroom on nuanced framing and long descriptions.

### Alternative transformer configurations

Several transformer and target variants were tested. Configurations that produced unstable loss, excessive truncation, or worse out-of-sample agreement were rejected. Expanding the target to count implicit social context also reduced precision and was abandoned.

### Selected model

ModernBERT-large was retained because it offered the strongest balance of classification quality and scalable inference. The selected configuration uses:

- Maximum sequence length of 384 tokens
- Dynamic padding and length-grouped batches
- Mixed-precision inference on GPU
- Validation-selected decision thresholds
- Three independently seeded models averaged at inference

The exact training and ensemble evaluation code is in `training/`.

## Evaluation policy

The decision threshold is chosen on the validation set and applied unchanged to the test set. This prevents optimistic performance from tuning directly on the headline test sample.

The ensemble evaluation additionally reports precision, recall, F1, confusion matrices, bootstrap confidence intervals, and performance at the validation-selected threshold. The test-set optimum is retained only as a clearly marked optimistic comparison.

High-confidence disagreements are exported for error analysis. This separates likely model failures from contested or ambiguous reference cases.

## Production inference

The production scorer preserves the ensemble probability calculation used during evaluation while adding stable Snowflake hash sharding, resume-safe exclusion of already-scored rows, length-sorted dynamic padding, shared device tensors across all three models, fused attention where supported, optional compilation, background prefetch, and asynchronous writes.

Every output row contains the person identifier, ensemble probability, binary label, model version, and scoring timestamp.

## Analysis design

The Snowflake layer retains the immutable score table and builds separate person- and education-level analytical bases. Aggregate outputs cover occupation, industry, employer, education, institution, demographics, and geography.

The analysis emphasizes robustness rather than raw rankings alone:

- Employer and demographic comparisons are repeated within broad occupation.
- College results report both person-weighted and institution-weighted estimates.
- Liberal-arts comparisons are stratified by selectivity.
- Institution matching is limited to unique normalized names.
- Summary length is treated as a major measurement concern because longer profiles offer more chances to express prosocial language.

These outputs are descriptive associations in model-readable text. They are not causal estimates of social value.
