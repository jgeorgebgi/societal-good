"""
Production framing scorer (optimized): run the 3-seed ModernBERT v6 ensemble
over every US person in PDL_CLEAN.ROOT_PERSON and write probabilities to a NEW
append-only table.

Prediction math is byte-for-byte the same as export_test_probs.py (3 seeds,
MAX_LEN=384, softmax class-1, averaged, fp16 AMP) so F1 parity is preserved.
Everything else is tuned for throughput on a GPU-bound batch job:

  * length-sorted dynamic padding   -> minimal wasted compute on PAD tokens
  * one host->device copy per batch, all 3 models run on the resident tensor
  * SDPA fused attention            -> faster ModernBERT, numerically equivalent
  * tf32 + cudnn.benchmark + inference_mode
  * overlapped I/O: a reader thread prefetches the next Snowflake chunk and a
    writer thread flushes results while the GPU keeps scoring
  * horizontal sharding via HASH(person_id) so N GPU boxes can run at once,
    resume-safe (each only claims unscored rows)

Run on the GPU box where best_model_modernbert_v6_seed* live:
    python score_full_db.py --threshold 0.50
Multi-GPU (e.g. 4 boxes):
    python score_full_db.py --num-shards 4 --shard 0   # ... 1, 2, 3 elsewhere
"""
import os
import glob
import time
import queue
import argparse
import threading
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas

ROOT = Path(__file__).resolve().parent   # portable: load .env from beside this script
load_dotenv(ROOT / ".env")

# ---- inference config (matches export_test_probs.py) -----------------------
MAX_LEN = 384
SEED_GLOB = "best_model_modernbert_v6_seed*"
MODEL_VERSION = "modernbert_v6_ens3"
CHUNK_ROWS = 20_000          # rows pulled + written per Snowflake round trip

# ---- where US-ness lives on ROOT_PERSON ------------------------------------
# CONFIRM with DESCRIBE TABLE ROOT_PERSON. Lowercase full name matches the
# EDUCATION_LOCATION_COUNTRY == "united states" convention in analyze.py.
COUNTRY_COL = "location_country"
COUNTRY_VAL = "united states"

# ---- global perf knobs -----------------------------------------------------
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


def load_models(seed_dirs, device, use_compile=False):
    """Load the ensemble once, preferring fused SDPA attention."""
    models = []
    for d in seed_dirs:
        try:
            m = AutoModelForSequenceClassification.from_pretrained(d, attn_implementation="sdpa")
        except (ValueError, ImportError):
            m = AutoModelForSequenceClassification.from_pretrained(d)
        m = m.to(device).eval()
        if use_compile:
            # ~1.5x, numerically ~identical; dynamic=True avoids recompiles on varied seq lengths
            m = torch.compile(m, dynamic=True)
        models.append(m)
    return models


def score_chunk(texts, models, tok, device, use_amp, batch):
    """Ensemble class-1 probability for each text, original order preserved.

    Sorts by token length so each GPU batch pads only to its own longest
    sequence, then scatters results back. All 3 seeds run on the same resident
    tensor (one transfer, not three)."""
    enc = tok(texts, truncation=True, max_length=MAX_LEN)
    ids_all, am_all = enc["input_ids"], enc["attention_mask"]
    order = sorted(range(len(ids_all)), key=lambda i: len(ids_all[i]))
    probs = np.empty(len(texts), dtype=np.float32)
    inv_n = 1.0 / len(models)

    for s in range(0, len(order), batch):
        idx = order[s:s + batch]
        feats = tok.pad(
            {"input_ids": [ids_all[i] for i in idx],
             "attention_mask": [am_all[i] for i in idx]},
            return_tensors="pt",
        )
        input_ids = feats["input_ids"].to(device, non_blocking=True)
        attn = feats["attention_mask"].to(device, non_blocking=True)
        acc = torch.zeros(len(idx), device=device)
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            for m in models:
                lg = m(input_ids=input_ids, attention_mask=attn).logits
                acc += torch.softmax(lg.float(), -1)[:, 1]
        acc = (acc * inv_n).float().cpu().numpy()
        probs[np.asarray(idx)] = acc
    return probs


def ensure_scores_table(conn, table):
    """Create the output table if absent. Never DROP / CREATE OR REPLACE."""
    conn.cursor().execute(f"""
        CREATE TABLE IF NOT EXISTS {table} (
            PERSON_ID      STRING,
            ENSEMBLE_PROB  FLOAT,
            LABEL          NUMBER(1),
            MODEL_VERSION  STRING,
            SCORED_AT      TIMESTAMP_NTZ
        )
    """)


def stream_unscored(conn, source, table, country_col, min_chars, num_shards, shard):
    """Yield CHUNK_ROWS-sized DataFrames of US rows not yet scored."""
    shard_clause = ""
    if num_shards > 1:
        shard_clause = f"AND MOD(ABS(HASH(p.person_id)), {num_shards}) = {shard}"
    sql = f"""
        SELECT p.person_id, p.summary
        FROM {source} p
        WHERE p.summary IS NOT NULL
          AND LENGTH(p.summary) >= {min_chars}
          AND LOWER(p.{country_col}) = '{COUNTRY_VAL}'
          {shard_clause}
          AND NOT EXISTS (SELECT 1 FROM {table} s WHERE s.person_id = p.person_id)
    """
    cur = conn.cursor()
    cur.execute(sql)
    for frame in cur.fetch_pandas_batches():
        frame.columns = [c.upper() for c in frame.columns]
        for start in range(0, len(frame), CHUNK_ROWS):
            yield frame.iloc[start:start + CHUNK_ROWS]
    cur.close()


def prefetch(gen, depth=2):
    """Run `gen` on a background thread so the next Snowflake fetch overlaps GPU work."""
    q, DONE, err = queue.Queue(maxsize=depth), object(), {}

    def run():
        try:
            for x in gen:
                q.put(x)
        except Exception as e:           # surface reader-thread failures
            err["e"] = e
        finally:
            q.put(DONE)

    threading.Thread(target=run, daemon=True).start()
    while True:
        x = q.get()
        if x is DONE:
            break
        yield x
    if "e" in err:
        raise err["e"]


def sf_connect():
    return snowflake.connector.connect(
        user=os.environ["SNOWFLAKE_USER"],
        password=os.environ["SNOWFLAKE_PASSWORD"],
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        warehouse=os.environ.get("SNOWFLAKE_WAREHOUSE"),
        database=os.environ.get("SNOWFLAKE_DATABASE", "PDL_CLEAN"),
        schema=os.environ.get("SNOWFLAKE_SCHEMA", "BGI_2026_05"),
        role=os.environ.get("SNOWFLAKE_ROLE"),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.50,
                    help="prob >= threshold => LABEL 1 (your sweep landed ~0.46-0.54)")
    ap.add_argument("--batch", type=int, default=64, help="GPU micro-batch; raise until VRAM-bound")
    ap.add_argument("--source", default="ROOT_PERSON")
    ap.add_argument("--table", default="FRAMING_SCORES_V6",
                    help="NEW output table; append-only, never overwritten")
    ap.add_argument("--country_col", default=COUNTRY_COL)
    ap.add_argument("--min_chars", type=int, default=40)
    ap.add_argument("--num-shards", type=int, default=1, help="total shards across all GPU boxes")
    ap.add_argument("--shard", type=int, default=0, help="this box's shard id (0..num-shards-1)")
    ap.add_argument("--limit_chunks", type=int, default=0, help=">0 to score only N chunks (smoke test)")
    ap.add_argument("--seeds", type=int, default=0,
                    help="use only first N seed models (0=all 3); 1 = single-seed, ~3x faster, ~1 F1 pt lower")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the models (~1.5x, no accuracy change)")
    args = ap.parse_args()

    read_conn = sf_connect()
    write_conn = sf_connect()          # separate connection so writes don't block reads
    ensure_scores_table(read_conn, args.table)

    seed_dirs = sorted(glob.glob(SEED_GLOB))
    assert seed_dirs, f"no model dirs match {SEED_GLOB}"
    if args.seeds > 0:
        seed_dirs = seed_dirs[:args.seeds]
    model_version = f"modernbert_v6_ens{len(seed_dirs)}"   # records how many seeds actually scored the row
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    tok = AutoTokenizer.from_pretrained(seed_dirs[0])
    models = load_models(seed_dirs, device, args.compile)
    print(f"device={device}  seeds={seed_dirs}  table={args.table}  "
          f"shard={args.shard}/{args.num_shards}  batch={args.batch}")

    writer = ThreadPoolExecutor(max_workers=1)   # one in-flight write, overlaps with GPU
    pending, total = [], 0

    def flush(df):
        write_pandas(write_conn, df, args.table, quote_identifiers=True)
        return len(df)

    stream = stream_unscored(read_conn, args.source, args.table, args.country_col,
                             args.min_chars, args.num_shards, args.shard)
    prev_t = time.time()
    steady_rows, steady_t = 0, None       # exclude chunk 0 (query/warehouse latency) from the rate
    for i, chunk in enumerate(prefetch(stream)):
        if args.limit_chunks and i >= args.limit_chunks:
            break
        t_gpu = time.time()
        probs = score_chunk(chunk["SUMMARY"].tolist(), models, tok, device, use_amp, args.batch)
        gpu_s = time.time() - t_gpu
        out = chunk[["PERSON_ID"]].reset_index(drop=True)   # clean RangeIndex -> no write_pandas index warning
        out["ENSEMBLE_PROB"] = np.round(probs, 4)
        out["LABEL"] = (probs >= args.threshold).astype(int)
        out["MODEL_VERSION"] = model_version
        out["SCORED_AT"] = datetime.now(timezone.utc).replace(tzinfo=None)

        pending = [f for f in pending if not f.done() or f.result() is None]
        pending.append(writer.submit(flush, out))
        total += len(out)
        now = time.time(); dt = now - prev_t; prev_t = now
        if i == 1:
            steady_t = now                # start the steady-state clock after the warm-up chunk
        elif i > 1:
            steady_rows += len(out)
        print(f"chunk {i}: {len(out)} rows in {dt:.1f}s "
              f"(gpu {gpu_s:.1f}s, {len(out)/max(dt,1e-9):.0f} rows/s); cumulative {total}")
    if steady_t is not None and steady_rows:
        rate = steady_rows / max(time.time() - steady_t, 1e-9)
        eta_h = 29_695_839 / (rate * max(args.num_shards, 1)) / 3600
        print(f"STEADY-STATE: {rate:.0f} rows/s/shard  ->  full 29.7M on {args.num_shards} shard(s) "
              f"~{eta_h:.1f} h")

    for f in pending:        # drain writes + surface any error
        f.result()
    writer.shutdown(wait=True)
    read_conn.close()
    write_conn.close()
    print(f"done. scored {total} new rows into {args.table}")


if __name__ == "__main__":
    main()
