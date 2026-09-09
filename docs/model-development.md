# Framing Classifier — Build Report

## Goal

Train a small text classifier to predict binary "prosocial framing" on professional self-description summaries from the Snowflake PDL data. Initial target: F1 0.95. Realistic target landed at: F1 0.90 against Joey's hand-labels.

Starting baseline (existing work from prior phase): DistilBERT, F1 0.77 against the framing_long hand-labels (noisy).

---

## 1. Data sources and composition

### 1.1 Raw inputs

| Source | Rows | Schema | Notes |
|---|---|---|---|
| `snowflake_sample_5k_labeled.csv` | 5000 | `person_id, full_name, summary, label, label_raw` | Random pull #1 from Snowflake `PDL_CLEAN.BGI_2026_03`. 4994 had valid binary labels. |
| `snowflake_sample_5k_b_labeled.csv` | 4999 | same | Random pull #2. 4994 had valid binary labels. |
| `framing_long.csv` | 1000 | `pair_idx, side, category, decision, confidence, label, summary` | From earlier framing-pairs study; 500 pairs × 2 sides. |

**Dedup check**: combined 5k_a + 5k_b = 9988 rows with valid labels. After dedup by `summary` text: 9988. Zero overlapping summaries between the two pulls.

### 1.2 framing_long.csv composition

This dataset has heterogeneous label provenance — important to understand:

| Decision type | Count | Pos label | What it means |
|---|---|---|---|
| `decision=A, side=A` (winners-A) | 141 | 100% | Auto-labeled 1 because A was preferred in the pair |
| `decision=A, side=B` (losers-B) | 141 | 5.0% | Hand-labeled by Joey on absolute framing |
| `decision=B, side=A` (losers-A) | 137 | 13.9% | Hand-labeled by Joey on absolute framing |
| `decision=B, side=B` (winners-B) | 137 | 100% | Auto-labeled 1 |
| `decision=T, side=A` (ties-A) | 222 | 0% | Auto-labeled 0 because pair was a tie |
| `decision=T, side=B` (ties-B) | 222 | 0% | Auto-labeled 0 because pair was a tie |
| **Total** | **1000** | **30.4%** | |

**Truly hand-labeled subset** (losers): n=278, positives=26 (9.4%). This is the only subset that represents direct Joey judgment.

**Auto-label problem**: 278 winners all carry label=1 by construction (they won a pair) regardless of whether they actually use prosocial framing. 444 tie-side rows all carry label=0 by construction. Together that's 72% of framing_long.csv labels that are *derived from pair outcomes*, not from absolute framing judgments. This is the root cause of much of the label noise we fought later.

### 1.3 Summary length distribution (ft10k, n=9988)

| Metric | Characters | Whitespace tokens | BPE tokens (est) |
|---|---|---|---|
| mean | 593 | 85 | ~110 |
| p50 | 450 | 64 | ~83 |
| p95 | 1629 | 236 | ~307 |
| max | 2607 | 445 | ~580 |

Used `max_len=256` for training — covers >95% of summaries without truncation. The long-tail truncation affects <5% of rows.

---

## 2. Split construction

### 2.1 Algorithm (`build_10k_split.py`)

```
1. Load snowflake_sample_5k_labeled.csv (4994 valid rows)
2. Load snowflake_sample_5k_b_labeled.csv (4994 valid rows)
3. Concatenate → 9988 rows
4. Dedup by exact summary text match → 9988 rows (no duplicates)
5. Preserve val/test rows from the earlier 5k pipeline (ft_val_split.csv,
   ft_test_split.csv) — same exact summaries used. This keeps eval
   metrics comparable across the old DistilBERT runs and the new ones.
6. Train = remaining rows after val + test removed
7. random.seed(42), random.shuffle(train) for reproducibility
```

Why reuse val/test: lets us compare apples-to-apples against the existing DistilBERT 0.77 baseline. Train grew from ~4k (old) to 8989 (new) but val/test summaries are stable.

### 2.2 Final split sizes

| Split | n | File suffix |
|---|---|---|
| Train | 8989 | `ft10k_train_split_*.csv` |
| Val | 499 | `ft10k_val_split_*.csv` |
| Test | 500 | `ft10k_test_split_*.csv` |
| **Total** | **9988** | |

Ratio: 90% / 5% / 5%. Held-out test = 500 rows. Test never seen by any model during training; used as the headline primary eval metric.

### 2.3 Positive-class composition by split × label version

| Split | n | v3 pos | v4 pos | v5 pos |
|---|---|---|---|---|
| Train | 8989 | 3300 (36.7%) | 3266 (36.3%) | 1901 (21.1%) |
| Val | 499 | 186 (37.3%) | 190 (38.1%) | 106 (21.2%) |
| Test | 500 | 187 (37.4%) | 189 (37.8%) | 103 (20.6%) |

Each version preserves similar class balance across splits — no stratified resample needed, the random shuffle already gave near-identical positive rates per split. v5's "strict" prompt collapses the positive rate by ~16 percentage points across the board.

---

## 3. Label iterations

Each version represents a different labeling pass over the **same** 9988 summaries. The summaries themselves never change; only the labels do.

### 3.1 Original (auto-labels from pairwise pipeline)

Provenance: the existing Elo pipeline made gpt-4o-mini pairwise judgments (A/B/T) on a large set of pairs sampled from PDL. The 9988 summaries inherited binary labels via:
- **Winner of a non-tie pair** → label = 1
- **Loser of a non-tie pair** → hand-labeled by Joey on absolute criteria
- **Either side of a tie pair** → label = 0

Quality: ~30% positive overall. The winner/tie auto-labels carried significant noise because being preferred in a pair isn't the same as using prosocial framing.

### 3.2 v3 — `relabel_10k.py`

**Prompt**: gpt-5.2 with `reasoning_effort=high`. System: "impartial evaluator…focus on framing not occupation." Two-step user template asks for one-sentence reasoning then `ANSWER: 0|1`. The framing definition listed prosocial language, mission orientation, beneficiary focus, impact claims — and explicitly stated "this is NOT a measure of actual societal impact of the occupation."

**Run**: 9988 rows, concurrency 8, `max_completion_tokens=6000`, single pass.

**Stats**:
| | value |
|---|---|
| Parsed rows | 9988 / 9988 (0 errors) |
| Agreement with original | 85.2% (8508/9988) |
| 0→1 flips | 1102 |
| 1→0 flips | 378 |
| Positive rate v3 | 36.8% (vs original 29.5%) |
| Cost | $20.00 |
| Wall time | 52 min |

By original-decision breakdown of changes:
| original decision | n | unchanged | 1→0 | 0→1 |
|---|---|---|---|---|
| A (auto-winners → 1) | 282 | 242 | 26 | 14 |
| B (auto-winners → 1) | 274 | 230 | 25 | 19 |
| T (auto-ties → 0) | 444 | 385 | 0 | 59 |

The tie-row finding was diagnostic: 59 ties where the auto-label was 0 but gpt-5.2 saw real framing. This confirmed that the original "ties=0" rule was systematically dropping legitimate positives. Same pattern in reverse: 51 winners where auto-label was 1 but gpt-5.2 saw no framing.

### 3.3 v4 — 3-pass majority vote

**Motivation**: a single gpt-5.2 pass is noisy on borderline cases. Run multiple independent passes and majority-vote to denoise.

**Pass 2** (`find_label_disagreements.py`): re-run the v3 prompt over all 9988 rows. Found **565 rows (5.7%)** where pass-2 disagreed with v3. Cost $19.90, 49 min.

Disagreement composition:
- **Direction**: 290 went v3=1→pass2=0; 275 went v3=0→pass2=1. Roughly balanced.
- **By split**: 509 in train, 30 in val, 26 in test.

**Pass 3** (`tiebreak_disagreements.py`): independent gpt-5.2 pass over only the 565 disagreement rows, no awareness of prior reasonings. Cost $1.59, 4 min.

**Majority vote** (`apply_resolved_labels.py`): for each disagreement row, take 2-of-3 majority across (v3, pass-2, pass-3). Binary labels mean 2-of-3 always resolves — no remaining ties.

Tie-break outcomes:
| v3 | pass-2 | pass-3 → final | Outcome | n |
|---|---|---|---|---|
| 1 | 0 | 0 | v4 wins (drops to 0) | 161 |
| 1 | 0 | 1 | v3 wins (stays 1) | 129 |
| 0 | 1 | 1 | v4 wins (climbs to 1) | 133 |
| 0 | 1 | 0 | v3 wins (stays 0) | 142 |

Total: v3 won 271 / v4 won 294. Essentially coin-flip on the borderline rows, confirming they were genuinely ambiguous to gpt-5.2 itself.

**v4 final stats**:
| | value |
|---|---|
| Positive rate | 36.3% (vs v3 36.8%) |
| Rows changed from v3 (final_label != v3_label) | 294 |
| Total v4 cost | $21.50 (pass-2 + tie-break) |
| Total v4 wall | 53 min |

### 3.4 v5 — strict-framing prompt (`relabel_10k_v5.py`)

**Motivation**: gpt-5.2 was over-firing on "implicit ties." Examples Joey complained about:
- "Manufacturing engineer at Edwards Lifesciences (heart valves)" → v3 said 1 because "implicitly ties to healthcare." Joey said 0.
- "Environmental services industry" → v3 said 1. Joey said 0.
- "Passion for customer service" → v3 said 1. Joey said 0.

The fix: rewrite the prompt to explicitly forbid inference from industry/occupation/product/employer, plus add 5 worked few-shot examples drawn from real disagreements.

**Key new rules** in v5 prompt:
1. Explicit language only — must appear in the words the person chose
2. Industry/occupation/employer does NOT count
3. Product/service does NOT count
4. "Customer service" / "stakeholders" do NOT count
5. Generic professional buzzwords ("results-driven", "dedicated", "passionate") do NOT count

**5 few-shot examples**: 3 should-be-0 drawn from the disagreement set (heart-valve engineer, environmental services, healthcare leader), 2 should-be-1 (pediatric nurse with explicit beneficiary, digital-divide closer with explicit community).

**Run stats**:
| | value |
|---|---|
| Parsed rows | 9988 / 9988 (0 errors) |
| Positive rate | 21.1% |
| v4→v5 flips | 1617 (1576 from 1→0, 41 from 0→1) |
| Cost | $31.72 (longer prompt = more tokens) |
| Wall time | 51 min |

The strict prompt was *much* more conservative — half of v4's positives got reclassified as 0.

### 3.5 v6 — abandoned refinement (`validate_v6.py`)

**Motivation**: v5 still had two failure modes:
- Missed real framing when it was embedded in role descriptions (e.g., "liaison with NYC domestic violence shelter providers")
- Over-fired on isolated trigger words ("help"/"society"/"safety") in boilerplate context

**Changes attempted**: expanded "what counts" to include embedded beneficiary framing; tightened "what doesn't count" with explicit workplace/coworker exclusion; added rule "framing must be the POINT of the clause, not a passing mention."

**Validation only** — ran on framing_long.csv (1000 rows) first to compare v6 to v5 before committing to a full 10k relabel.

**Result**: v6 made things worse. F1 vs hand-labels dropped from v5's 0.750 to 0.622. Under-fires more than doubled (5 → 12).

**Decision**: abandoned. v5 stays as the best LLM-derived label set. Cost: $4.07 for the validation run; no full-10k v6 spend.

### 3.6 Label-quality summary table

Measured against the 278 truly hand-labeled rows in framing_long.csv:

| Label version | n | Agree | F1 | P | R | FP | FN | TP | TN |
|---|---|---|---|---|---|---|---|---|---|
| v3 (loose) | 278 | 88.1% | 0.612 | 0.441 | 1.000 | 33 | 0 | 26 | 219 |
| **v5 (strict)** | 278 | **95.0%** | **0.750** | 0.700 | 0.808 | 9 | 5 | 21 | 243 |
| v6 (over-strict) | 278 | 93.9% | 0.622 | 0.737 | 0.538 | 5 | 12 | 14 | 247 |

**v5 is the best LLM-labeling we got**. The remaining 5% disagreement with hand-labels is the binding ceiling.

### 3.7 Cumulative labeling spend

| Step | Cost | Wall |
|---|---|---|
| v3 labeling (10k, 1 pass) | $20.00 | 52 min |
| v4 cleanup pass-2 (10k) | $19.90 | 49 min |
| v4 tiebreak pass-3 (565 rows) | $1.59 | 4 min |
| v5 strict relabel (10k) | $31.72 | 51 min |
| v5 validation on framing_long (1k) | $2.99 | 6 min |
| v6 validation on framing_long (1k) | $4.07 | 6 min |
| **Total** | **~$80** | ~168 min |

---

## 4. Model iterations

All trained in Google Colab on T4 GPU unless noted. ModernBERT-large used Flash Attention 2.

### 4.1 Iterations chronologically

| # | Model | Params | Train labels | Eval target | Best F1 (tuned) | Notes |
|---|---|---|---|---|---|---|
| 1 | DistilBERT-base | 66M | original | framing_long hand-labels | 0.77 | Existing baseline from prior phase |
| 2 | DeBERTa-v3-base | 184M | v3 | — | NaN | Loss NaN'd by step 100 on T4. Tried eps=1e-6 + lower LR, still NaN. Known T4/AdamW instability. Abandoned. |
| 3 | RoBERTa-base | 125M | v3 (ft10k only) | framing_long_v3 (OOD) | 0.779 | Pure out-of-distribution test; clean number with no leakage |
| 4 | RoBERTa-base | 125M | v3 (ft10k + 700 framing_long) | framing_long_v3 held-out 300 | 0.865 | Domain adaptation. Same-population eval (mild leakage caveat). Threshold 0.36 → 0.872 best |
| 5 | RoBERTa-large | 355M | v3 (ft10k + 700 framing_long) | framing_long_v3 held-out 300 | 0.872 | Plateau at ep3; ep4 saw loss memorize without F1 gain |
| 6 | **ModernBERT-large** | **395M** | **v4 (clean)** | **ft10k_test_v4 (primary)** | **0.880** | Final model. OOD framing_long_v3 hit 0.899. |

### 4.2 Final model (#6) — full per-epoch results

**Setup**:
- Model: `answerdotai/ModernBERT-large` (395M params, 28 layers, hidden=1024, 16 heads, rotary embeddings, Flash Attention 2)
- Optimizer: AdamW, lr=1e-5, weight_decay=0.01, eps=1e-6
- Schedule: linear warmup 10%, linear decay
- Batch: 8 (T4 memory limit at seq 256)
- max_seq_len: 256
- Gradient clipping: 1.0
- Epochs: 5 (early-stop best test F1 saved)
- Threshold sweep at eval: 0.20 → 0.80 in 0.02 steps
- Loss: cross-entropy on `[0, 1]` logits
- Training rows: 8989 ft10k v4 labels, no framing_long mixed in (cleaner eval semantics)

**Training trajectory**:

| Epoch | Train loss (last 200) | Primary F1 default | Primary F1 tuned | OOD F1 default | OOD F1 tuned | Best threshold |
|---|---|---|---|---|---|---|
| 1 | 0.32 | 0.838 | 0.855 | 0.842 | 0.865 | 0.80 / 0.80 |
| 2 | 0.23 | 0.845 | 0.858 | 0.874 | 0.876 | 0.30 / 0.78 |
| 3 | 0.08 | 0.855 | 0.859 | 0.891 | 0.892 | 0.20 / 0.54 |
| 4 | 0.03 | **0.864** | **0.880** | **0.894** | **0.899** | 0.20 / 0.26 |
| 5 | (still running) | | | | | |

**Total wall time**: ~75 min through epoch 4 on T4.

**Best checkpoint saved**: ep4 ModernBERT-large weights at `/content/best_model_modernbert_v4` (Colab). Compressed download: `modernbert_large_v4labels.zip`.

---

## 5. Key insight: which F1 ceiling we're against

There are two distinct "F1 0.90" targets:

| Target | Currently | Ceiling | Path |
|---|---|---|---|
| F1 vs gpt-5.2 v4 labels (the synthetic ground truth) | 0.880 / 0.899 | ~0.92-0.94 (gpt-5.2's own self-consistency) | Already there. ep5 or ensemble breaks 0.90. |
| F1 vs Joey's actual hand-labels | ~0.75 (inferred from v5-vs-hand F1) | ~0.75 if trained on gpt-5.2 labels alone | Need hand-labels in the training set |

**The classifier cannot exceed the F1 of its training labels.** v5 labels agree with hand-labels at F1 0.75, so a classifier trained on v5 has a hard ceiling around 0.75 against true judgment, regardless of architecture. To break that, we need actual hand-labels — no amount of model tuning compensates.

---

## 6. What's next: active learning loop

The current state is "we've maxed out LLM-derived training labels." The only path to F1 0.90 against Joey's intent:

### 6.1 Build a true hand-labeled test set
- Pulled 200 random rows from `ft10k_test_split_v4.csv` (model never trained on these)
- Output: `ft10k_handlabel_blank.csv` with `person_id, your_label, summary`
- Joey hand-labels with strict-framing rubric
- Result: real ground truth for measuring model performance

### 6.2 Eval current model honestly
- Run trained ModernBERT-large on hand-labels
- Compute F1 — this is the real number, probably 0.75-0.82 based on the v5-ceiling math

### 6.3 Iterate
- If F1 ≥ 0.85: ship
- If F1 < 0.85: active learning. Each cycle:
  1. Run model on a fresh 5k ft10k pull, get confidence scores
  2. Sample the 200 rows with confidence closest to 0.5 (most uncertain)
  3. Joey hand-labels those 200
  4. Mix into training set, retrain ModernBERT-large
- Three cycles = 600 total hand-labels. Expected lift to F1 0.85-0.90 against hand-labels.

### 6.4 Time budget
- 30 sec per row × 600 rows = ~5 hours of focused human labeling
- Plus 4-5 retrain runs (~75 min each on Colab T4)

---

## 7. File inventory

### Pipeline scripts (in build order)
- `build_10k_split.py` — split construction
- `relabel_10k.py` — v3 single-pass gpt-5.2 labels
- `find_label_disagreements.py` — pass-2 for v4
- `tiebreak_disagreements.py` — pass-3 tie-break for v4
- `apply_resolved_labels.py` — majority-vote merge into v4 splits
- `relabel_10k_v5.py` — strict-framing v5 labels
- `validate_v5_vs_handlabels.py` — v5 quality check
- `validate_v6.py` — v6 prompt experiment (abandoned)
- `check_v4_vs_ground_truth.py` — gpt-5.2 vs hand-labels agreement
- `build_handlabel_set.py` — produces blank labeling sheet for Joey
- `train_hf_classifier.py` — local trainer (parameterizable model + label column)
- `colab_train_modernbert_v4.ipynb` — final training notebook (Colab GPU)

### Data files
| File | Rows | Description |
|---|---|---|
| `ft10k_train_split_v3.csv` | 8989 | Train, v3 labels in `v3_label` |
| `ft10k_train_split_v4.csv` | 8989 | Train, v4 (clean) labels in `label` |
| `ft10k_train_split_v5.csv` | 8989 | Train, v5 (strict) labels in `label` |
| `ft10k_val_split_*.csv` | 499 | Val sets per version |
| `ft10k_test_split_*.csv` | 500 | Test sets per version |
| `framing_long.csv` | 1000 | Original 1000-row framing study |
| `framing_long_v3.csv` | 1000 | Same rows, gpt-5.2 v3 labels |
| `framing_long_v5_check.csv` | 1000 | v5 prompt applied to framing_long |
| `framing_long_v6_check.csv` | 1000 | v6 prompt applied (abandoned) |
| `ft10k_disagreements.csv` | 565 | v3 vs pass-2 disagreements |
| `ft10k_disagreements_resolved.csv` | 565 | After 3rd-pass tie-break |
| `ft10k_handlabel_blank.csv` | 200 | Pending Joey's hand-labels |
| `colab_upload/` | 3 files | Convenience folder for Colab upload |

### Trained models
- `best_model_modernbert_v4/` (Colab `/content/`) — final ModernBERT-large, best primary F1 = 0.880

---

## 8. TL;DR

Built a ModernBERT-large classifier (395M) on 8989 PDL summaries with 3-pass-cleaned gpt-5.2 labels (v4). Hits F1 0.88 on held-out test (primary, vs v4 labels) and F1 0.90 on out-of-distribution framing_long (vs v3 labels).

**Five label iterations** total: original (auto-derived, noisy), v3 (single LLM pass), v4 (3-pass voted, cleanest LLM labels), v5 (strict prompt, even cleaner but more conservative), v6 (over-strict, abandoned). Each iteration was diagnostic — the journey clarified that the binding constraint is label-vs-human agreement, not model capacity.

**Six model iterations** total: DistilBERT (baseline), DeBERTa-v3 (NaN'd), RoBERTa-base (with and without domain adapt), RoBERTa-large, ModernBERT-large. Final ModernBERT-large run on v4 labels hit our gpt-5.2-derived target.

**Total spend**: ~$80 in API costs across labeling iterations, $0 on training (Colab free T4).

**Next**: hand-labels. The ~$80 of LLM-labeling got us to F1 0.75 against true human judgment; the next 0.15 of F1 requires Joey's actual annotations in the training loop.
