"""
Export the 3-seed ensemble's probability for every v6-test row to a small CSV.
Run on the EC2 box (where the models live), then scp the tiny output down — that's
all you need locally to score the model against the reference evaluation labels.

Run: python export_test_probs.py
Output: ./test_ensemble_probs.csv  (person_id, ensemble_prob)  — a few KB
"""
import csv, glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding

MAX_LEN, BATCH = 384, 16
SEED_DIRS = sorted(glob.glob('best_model_modernbert_v6_seed*'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
use_amp = device.type == 'cuda'
print(f'models = {SEED_DIRS}')

tok = AutoTokenizer.from_pretrained(SEED_DIRS[0])
collator = DataCollatorWithPadding(tok, padding='longest', max_length=MAX_LEN)

rows = [r for r in csv.DictReader(open('ft10k_test_split_v6.csv'))
        if r['summary'].strip() and r.get('v6_label') in ('0', '1')]
X = [r['summary'] for r in rows]


class DS(Dataset):
    def __init__(self, X): self.enc = tok(X, truncation=True, max_length=MAX_LEN)
    def __len__(self): return len(self.enc['input_ids'])
    def __getitem__(self, i):
        return {'input_ids': self.enc['input_ids'][i], 'attention_mask': self.enc['attention_mask'][i]}


loader = DataLoader(DS(X), batch_size=BATCH, shuffle=False, collate_fn=collator)  # order preserved
stack = []
for d in SEED_DIRS:
    m = AutoModelForSequenceClassification.from_pretrained(d).to(device); m.eval()
    out = []
    with torch.no_grad():
        for b in loader:
            ids, am = b['input_ids'].to(device), b['attention_mask'].to(device)
            with torch.amp.autocast('cuda', dtype=torch.float16, enabled=use_amp):
                lg = m(input_ids=ids, attention_mask=am).logits
            out.extend(torch.softmax(lg.float(), -1)[:, 1].cpu().tolist())
    stack.append(out); del m
    if use_amp: torch.cuda.empty_cache()
prob = np.mean(stack, axis=0)

with open('test_ensemble_probs.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['person_id', 'ensemble_prob'])
    for r, p in zip(rows, prob):
        w.writerow([r['person_id'], round(float(p), 4)])
print(f'wrote test_ensemble_probs.csv  ({len(rows)} rows)')
