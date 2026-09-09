"""
Error analysis: where does the 3-seed ensemble DISAGREE with the v6 test label?
Goal: tell apart model mistakes from LABEL mistakes.

For every disagreement it dumps the model's confidence, the gold v6 label, the v6
confidence (1.0=4-0, 0.75=3-1, 0.50=2-2 split), the 4 individual ensemble votes, and
each voter's reasoning for structured review of the reference label.

Sorted so the most confident misses come first (model very sure + disagrees = label most suspect).

Run: python analyze_errors_v6.py
Output: ./errors_v6_review.csv  (open in Excel/Sheets)  + a console summary.
"""
import csv, glob, json, os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding

MAX_LEN = 384
BATCH = 16
TEST_CSV = 'ft10k_test_split_v6_final.csv' if os.path.exists('ft10k_test_split_v6_final.csv') else 'ft10k_test_split_v6.csv'
LABEL_COL = 'v6_final' if TEST_CSV.endswith('final.csv') else 'v6_label'
SEED_DIRS = sorted(glob.glob('best_model_modernbert_v6_seed*'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
use_amp = device.type == 'cuda'

# threshold from the training run (fall back to 0.5)
THRESH = 0.5
if os.path.exists('ensemble_v6.json'):
    THRESH = json.load(open('ensemble_v6.json')).get('threshold', 0.5)
print(f'test={TEST_CSV}  label={LABEL_COL}  models={SEED_DIRS}  threshold={THRESH}')

tok = AutoTokenizer.from_pretrained(SEED_DIRS[0])
collator = DataCollatorWithPadding(tok, padding='longest', max_length=MAX_LEN)

# Keep full rows so we can report votes + reasoning; preserve file order so probs align.
rows = [r for r in csv.DictReader(open(TEST_CSV)) if r['summary'].strip() and r.get(LABEL_COL) in ('0', '1')]
X = [r['summary'] for r in rows]
y = np.array([int(r[LABEL_COL]) for r in rows])


class DS(Dataset):
    def __init__(self, X):
        self.enc = tok(X, truncation=True, max_length=MAX_LEN)
    def __len__(self): return len(self.enc['input_ids'])
    def __getitem__(self, i):
        return {'input_ids': self.enc['input_ids'][i], 'attention_mask': self.enc['attention_mask'][i]}


loader = DataLoader(DS(X), batch_size=BATCH, shuffle=False, collate_fn=collator)  # order preserved


def probs(model):
    model.eval(); out = []
    with torch.no_grad():
        for b in loader:
            ids, am = b['input_ids'].to(device), b['attention_mask'].to(device)
            with torch.amp.autocast('cuda', dtype=torch.float16, enabled=use_amp):
                lg = model(input_ids=ids, attention_mask=am).logits
            out.extend(torch.softmax(lg.float(), -1)[:, 1].cpu().tolist())
    return np.array(out)


stack = []
for d in SEED_DIRS:
    m = AutoModelForSequenceClassification.from_pretrained(d).to(device)
    stack.append(probs(m)); del m
    if use_amp: torch.cuda.empty_cache()
p = np.mean(stack, axis=0)                 # ensemble probability, aligned to rows
pred = (p >= THRESH).astype(int)

VOTES = ['v54_strict_label', 'v54_procon_label', 'v5_label', 'v54mini_strict_label']
REASONS = ['v54_strict_reasoning', 'v54_procon_reasoning', 'v5_reasoning', 'v54mini_strict_reasoning']

disagree = np.where(pred != y)[0]
# confidence the model has in its own (disagreeing) call: distance of prob from threshold
margin = np.where(pred == 1, p - THRESH, THRESH - p)
order = disagree[np.argsort(-margin[disagree])]   # most confident misses first

out_rows = []
for i in order:
    r = rows[i]
    out_rows.append({
        'person_id': r['person_id'],
        'model_prob': round(float(p[i]), 3),
        'model_pred': int(pred[i]),
        'v6_label': int(y[i]),
        'v6_confidence': r.get('v6_confidence', ''),
        'miss_type': 'false_neg(model=0,label=1)' if pred[i] == 0 else 'false_pos(model=1,label=0)',
        **{v: r.get(v, '') for v in VOTES},
        'summary': r['summary'],
        **{v: r.get(v, '') for v in REASONS},
    })

with open('errors_v6_review.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
    w.writeheader(); w.writerows(out_rows)

# ---- console summary ----
print(f'\n{len(disagree)}/{len(y)} disagreements at t={THRESH}  (acc={1-len(disagree)/len(y):.3f})')
print('\nDisagreements by v6 label confidence (low conf = shaky label, likely a label error):')
for c in ('0.5', '0.50', '0.75', '1.0', '1.00'):
    sel = [i for i in disagree if rows[i].get('v6_confidence') == c]
    if sel: print(f'  v6_confidence={c}: {len(sel)} of {len(disagree)} misses')
hi = [i for i in disagree if margin[i] > 0.40]
print(f'\nHigh-confidence misses (model >0.40 past threshold): {len(hi)} -> most likely LABEL errors')
unan = [i for i in disagree if len(set(rows[i].get(v) for v in VOTES)) > 1]
print(f'Misses where the 4 voters were NOT unanimous: {len(unan)} of {len(disagree)} -> label was contested')
print('\nWrote errors_v6_review.csv  (sorted: most-confident misses first).')
