"""
Merge your hand-reviewed split-decision labels back into the v6 test file.

Workflow:
  1. Open v6_splits_for_review.csv. For each of the 13 split rows, fill
     `your_call` with 0 or 1 based on your reading of the summary + reasoning.
  2. Run: python3 apply_v6_splits_review.py

Output: ft10k_test_split_v6_final.csv with a v6_final column. v6_final = your_call
where you supplied one, else v6_label.
"""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

reviews = {}
for r in csv.DictReader(open(ROOT / "v6_splits_for_review.csv")):
    yc = r.get("your_call", "").strip()
    if yc in ("0", "1"):
        reviews[r["person_id"]] = int(yc)

print(f"Loaded {len(reviews)} hand-reviewed labels from v6_splits_for_review.csv")

rows = list(csv.DictReader(open(ROOT / "ft10k_test_split_v6.csv")))
fields = list(rows[0].keys())
if "v6_final" not in fields:
    fields.insert(fields.index("v6_label") + 1, "v6_final")

n_changed = 0
n_split_unresolved = 0
for r in rows:
    pid = r["person_id"]
    if pid in reviews:
        r["v6_final"] = str(reviews[pid])
        if r["v6_final"] != r["v6_label"]:
            n_changed += 1
    else:
        r["v6_final"] = r["v6_label"]
        if r.get("v6_confidence") == "0.50":
            n_split_unresolved += 1

out_path = ROOT / "ft10k_test_split_v6_final.csv"
with open(out_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)

n_final_pos = sum(1 for r in rows if r["v6_final"] == "1")
print(f"\nWrote {out_path.name}: {len(rows)} rows, {n_final_pos} positives "
      f"({n_final_pos/len(rows):.1%})")
print(f"v6_final overrides v6_label on: {n_changed} rows")
print(f"Split rows still unresolved (no your_call): {n_split_unresolved}")
