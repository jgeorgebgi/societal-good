"""
Combine snowflake_sample_5k_labeled.csv + snowflake_sample_5k_b_labeled.csv
into a 10k labeled set, dedup, stratified 80/10/10 split.
Reuses the original 5k val/test rows for continuity (so we can compare
classifiers fairly across runs).
"""
import csv, random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
random.seed(42)


def load_labeled(path):
    return [r for r in csv.DictReader(open(path)) if r.get("label") in ("0", "1")]


a = load_labeled(ROOT / "snowflake_sample_5k_labeled.csv")
b = load_labeled(ROOT / "snowflake_sample_5k_b_labeled.csv")
print(f"a={len(a)}  b={len(b)}")

# Dedup by summary text
seen = set()
combined = []
for r in a + b:
    s = r["summary"].strip()
    if s and s not in seen:
        seen.add(s)
        combined.append(r)
print(f"combined (deduped) = {len(combined)}")

# Reuse the original val/test to keep eval splits comparable
val_summaries = {r["summary"].strip() for r in csv.DictReader(open(ROOT / "ft_val_split.csv"))}
test_summaries = {r["summary"].strip() for r in csv.DictReader(open(ROOT / "ft_test_split.csv"))}
print(f"existing val={len(val_summaries)}  test={len(test_summaries)}")

# Build new train = everything that isn't in val/test, keep val/test as-is
val_rows = [r for r in combined if r["summary"].strip() in val_summaries]
test_rows = [r for r in combined if r["summary"].strip() in test_summaries]
train_rows = [r for r in combined if r["summary"].strip() not in val_summaries
                                      and r["summary"].strip() not in test_summaries]

random.shuffle(train_rows)
print(f"\nTrain: {len(train_rows)}  ({sum(1 for r in train_rows if r['label']=='1')} pos)")
print(f"Val:   {len(val_rows)}  ({sum(1 for r in val_rows if r['label']=='1')} pos)")
print(f"Test:  {len(test_rows)}  ({sum(1 for r in test_rows if r['label']=='1')} pos)")


def write(path, rows):
    fields = ["person_id", "label", "summary"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})
    print(f"  wrote {path}")


print()
write(ROOT / "ft10k_train_split.csv", train_rows)
write(ROOT / "ft10k_val_split.csv", val_rows)
write(ROOT / "ft10k_test_split.csv", test_rows)
