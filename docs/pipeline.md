# Pipeline and decision record

This is the curated path used for the production result. It intentionally omits superseded prompt variants, exploratory notebooks, local scratch files, raw profile exports, and duplicate model implementations.

## Flow

```text
Strict labeling rubric
        ↓
Deduplicated fixed splits (`pipeline/build_10k_split.py`)
        ↓
Four-voter evaluation labels (`pipeline/relabel_v6_ensemble.py`)
        ↓
Human review of 2–2 ties (`pipeline/apply_v6_splits_review.py`)
        ↓
ModernBERT training (`training/train_modernbert_v6_best.py`)
        ↓
Three-seed evaluation and threshold selection
(`training/evaluate_ensemble_v6.py`)
        ↓
Production sharded scoring (`src/score_full_db.py`)
        ↓
Snowflake analytical tables (`sql/create_analysis_tables.sql`)
        ↓
Hypothesis and robustness analyses (`analysis/`)
```

## Retained decisions

### Construct

The label measures explicit prosocial framing in a self-description. Occupation, employer, industry, and socially useful products do not qualify without explicit language about beneficiaries, mission, community, or public welfare. This prevents the label from silently becoming a subjective occupation ranking.

### Label quality

Earlier labels inherited pairwise winners and ties, which did not reliably represent an absolute binary construct. The final evaluation labels use four independent votes. Unanimous and 3–1 outcomes are retained; 2–2 splits are surfaced for manual review. The exact rules and examples are in `labeling-rubric.md`; the full iteration history is in `model-development.md`.

### Model choice

ModernBERT was selected after simpler baselines and alternate transformer configurations. The retained training script uses a 384-token limit, dynamic padding, length-grouped batches, mixed precision, and validation-selected thresholds. The production system averages three independently seeded models.

### Honest evaluation

The decision threshold is selected on validation data and applied unchanged to the test set. `training/evaluate_ensemble_v6.py` reports bootstrap intervals and distinguishes deployable test performance from optimistic test-tuned performance.

### Production scaling

`src/score_full_db.py` preserves the evaluation probability calculation while adding stable hash sharding, asynchronous Snowflake reads and writes, length sorting, fused attention, optional compilation, and resume-safe anti-joins. The table records the model version and scoring timestamp for every person.

### Analysis

The analysis preserves a person-level base and a deduplicated education base, then creates named aggregate tables. Robustness scripts compare groups within the same broad occupation where possible. The college analysis uses a conservative unique-name IPEDS match and reports person-weighted and institution-weighted results.

## Inputs intentionally not committed

- Raw professional-profile summaries and person identifiers
- Human- and model-labeled split CSVs containing profile text
- Snowflake credentials or API keys
- Model checkpoints and optimizer state
- Full PDL tables and intermediate extracts

The scripts document the required filenames and schemas. Authorized users can regenerate them from the governed data source without copying sensitive text into GitHub.
