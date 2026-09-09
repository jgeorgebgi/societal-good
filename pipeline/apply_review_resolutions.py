"""Merge resolved split-decision labels back into the v6 test file.

Workflow:
  1. Resolve the split rows in v6_splits_for_review.csv by setting
     `resolved_label` to 0 or 1 under the published label definition.
  2. Run: python3 pipeline/apply_review_resolutions.py

Output: ft10k_test_split_v6_final.csv with a v6_final column. v6_final uses
`resolved_label` where supplied and otherwise falls back to v6_label.
"""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

reviews = {}
for r in csv.DictReader(open(ROOT / "v6_splits_for_review.csv")):
    yc = r.get("resolved_label", "").strip()
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
print(f"Split rows still unresolved (no resolved_label): {n_split_unresolved}")
