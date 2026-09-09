# Societal Good Framing Model

Reproducible scoring and analysis for a ModernBERT ensemble that identifies explicit prosocial or societal-good framing in professional-profile text.

## What is here

- `src/score_full_db.py` — resumable, horizontally sharded GPU scoring over Snowflake data.
- `sql/create_analysis_tables.sql` — person-, education-, occupation-, industry-, college-, and geography-level analytical tables.
- `analysis/` — hypothesis tests and robustness checks, including the liberal-arts-college comparison.
- `results/` — aggregate outputs and charts used in the findings report. These contain no person-level profile text.
- `scripts/publish_results_to_snowflake.py` — optional loader that publishes each aggregate CSV as an easy-to-query Snowflake table.
- `docs/findings.md` — meeting-ready interpretation, caveats, and headline results.

Large model weights, raw profile text, credentials, caches, archives, and intermediate training files are intentionally excluded.

## Current production data

The production score table is:

```text
PDL_CLEAN.BGI_2026_05.FRAMING_SCORES_V6_SEED123
```

It contains 29,695,839 unique scored people. The analytical tables created from it use the `FRAMING_ANALYSIS_*` prefix in the same database and schema. See `sql/create_analysis_tables.sql` for the exact definitions.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Populate `.env` locally. Never commit it.

## Run the scorer

Model directories named `best_model_modernbert_v6_seed*` must sit beside the scoring script on each GPU machine. For four workers:

```bash
python src/score_full_db.py --table FRAMING_SCORES_V6_SEED123 --num-shards 4 --shard 0 --compile
python src/score_full_db.py --table FRAMING_SCORES_V6_SEED123 --num-shards 4 --shard 1 --compile
```

Run shards 2 and 3 analogously on the other workers. Sharding uses a stable Snowflake hash of `PERSON_ID`; writes are append-only and resume-safe.

## Rebuild the analyses

First run `sql/create_analysis_tables.sql` in Snowflake, then:

```bash
python analysis/run_comprehensive_hypotheses.py
python analysis/run_adjusted_hypotheses.py
python analysis/build_meeting_charts.py
```

Set `RESUME=1` to reuse saved CSVs instead of re-querying Snowflake.

## Publish aggregate CSVs to Snowflake

The production tables already contain the main scores and analytical bases. To expose every saved aggregate as a simple table:

```bash
python scripts/publish_results_to_snowflake.py --dry-run
python scripts/publish_results_to_snowflake.py --replace
```

Each file becomes `PDL_CLEAN.BGI_2026_05.SOCIETAL_GOOD_<FILE_NAME>`, with the CSV column names preserved in uppercase. A `SOCIETAL_GOOD_DATASETS` manifest records table names and row counts.

## Interpretation

Outputs are descriptive associations with what the model can read in profile text, not causal estimates of social contribution. Profile-summary length is a major confounder and should be adjusted for before publishing institutional rankings.
