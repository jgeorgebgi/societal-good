"""
Evaluate the saved 3-seed ensemble — no retraining, just loads the models and scores.
Reports, for each eval set:
  - honest F1   : threshold chosen on VAL, applied to that set (deployable)
  - own-best F1 : threshold tuned on the set itself (the optimistic, v4-notebook-style number)
  - 95% bootstrap CI on the honest F1 (so you know the noise band on ~500 rows)

Run: python evaluate_ensemble_v6.py
"""
import csv, os, glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from sklearn.metrics import precision_recall_fscore_support

MAX_LEN = 384
BATCH = 16
SEED_DIRS = sorted(glob.glob('best_model_modernbert_v6_seed*'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
use_amp = device.type == 'cuda'
print(f'device={device}  models={SEED_DIRS}')

tok = AutoTokenizer.from_pretrained(SEED_DIRS[0])
collator = DataCollatorWithPadding(tok, padding='longest', max_length=MAX_LEN)


def load(path, col):
    rows = [r for r in csv.DictReader(open(path)) if r['summary'].strip() and r.get(col) in ('0', '1')]
    return [r['summary'] for r in rows], [int(r[col]) for r in rows]


class DS(Dataset):
    def __init__(self, X, y):
        self.enc = tok(X, truncation=True, max_length=MAX_LEN); self.y = y
        self.lengths = [len(i) for i in self.enc['input_ids']]
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        return {'input_ids': self.enc['input_ids'][i],
                'attention_mask': self.enc['attention_mask'][i], 'labels': self.y[i]}


class LenSampler(Sampler):
    def __init__(self, lengths, bs): self.order = sorted(range(len(lengths)), key=lambda i: lengths[i]); self.bs = bs
    def __iter__(self):
        return iter([self.order[i:i+self.bs] for i in range(0, len(self.order), self.bs)])
    def __len__(self): return (len(self.order)+self.bs-1)//self.bs


def loader(ds):
    return DataLoader(ds, batch_sampler=LenSampler(ds.lengths, BATCH), collate_fn=collator)


def probs_for(model, ld):
    model.eval(); ps, ys = [], []
    with torch.no_grad():
        for b in ld:
            ys.extend(b['labels'].tolist())
            ids, am = b['input_ids'].to(device), b['attention_mask'].to(device)
            with torch.amp.autocast('cuda', dtype=torch.float16, enabled=use_amp):
                lg = model(input_ids=ids, attention_mask=am).logits
            ps.extend(torch.softmax(lg.float(), -1)[:, 1].cpu().tolist())
    return np.array(ps), np.array(ys)


SETS = {}
if os.path.exists('ft10k_test_split_v6_final.csv'):
    SETS['test'] = load('ft10k_test_split_v6_final.csv', 'v6_final')
else:
    SETS['test'] = load('ft10k_test_split_v6.csv', 'v6_label')
SETS['val'] = load('ft10k_val_split_v6.csv', 'v6_label')
SETS['ood_framing_long_v3'] = load('framing_long_v3.csv', 'v3_label')

# Average probabilities across the seed models (note: order of SETS is fixed, so labels align)
avg = {}
for name, (X, y) in SETS.items():
    ld = loader(DS(X, y))
    pstack = []
    for d in SEED_DIRS:
        m = AutoModelForSequenceClassification.from_pretrained(d).to(device)
        p, yy = probs_for(m, ld)
        pstack.append(p); del m
        if use_amp: torch.cuda.empty_cache()
    avg[name] = (np.mean(pstack, axis=0), np.array(yy))


def f1_at(p, y, t):
    _, _, f1, _ = precision_recall_fscore_support(y, (p >= t).astype(int), average='binary', pos_label=1, zero_division=0)
    return f1


GRID = np.round(np.arange(0.10, 0.96, 0.01), 2)   # finer + wider than the training sweep


def best_t(p, y):
    f1, t = max((f1_at(p, y, t), float(t)) for t in GRID)
    return t, f1


def boot_ci(p, y, t, n=1000):
    rng = np.random.default_rng(0); idx = np.arange(len(y)); out = []
    for _ in range(n):
        s = rng.choice(idx, len(idx), replace=True)
        out.append(f1_at(p[s], y[s], t))
    return np.percentile(out, 2.5), np.percentile(out, 97.5)


# threshold chosen on VAL, finer grid
t_val, f1_val = best_t(*avg['val'])
print(f'\nThreshold chosen on val (finer grid): t*={t_val:.2f}  (val F1={f1_val:.3f})\n')
print(f'{"set":24s} {"honest F1 (t* from val)":26s} {"own-best F1 (peeking)":22s}')
print('-'*74)
for name, (p, y) in avg.items():
    honest = f1_at(p, y, t_val)
    lo, hi = boot_ci(p, y, t_val)
    ob_t, ob = best_t(p, y)
    print(f'{name:24s} {honest:.3f}  [95% {lo:.3f}-{hi:.3f}]      {ob:.3f} @t={ob_t:.2f}')
print('\nHonest = deployable (threshold from val). Own-best = optimistic / v4-notebook-style.')
