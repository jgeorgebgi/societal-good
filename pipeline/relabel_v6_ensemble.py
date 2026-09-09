"""
4-vote thorough ensemble labeler for the ft10k test split.

Every row gets 4 independent votes (no tiebreaking — true ensemble):
  1. gpt-5.4 high reasoning, STRICT prompt
  2. gpt-5.4 high reasoning, PRO/CON prompt (considers counter-arguments)
  3. gpt-5.2 v5_label (reused from ft10k_test_split_v5.csv)
  4. gpt-5.4-mini high reasoning, STRICT prompt

v6_label = majority of 4.
v6_confidence ∈ {1.0 (4-0), 0.75 (3-1), 0.5 (2-2 split — review)}.

Cost ~ $95 for 1,474 rows (~25 min wall).
"""
import argparse
import asyncio
import csv
import re
import time
from pathlib import Path
from collections import Counter
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

DEFAULT_INP = "ft10k_test_split_v5.csv"
DEFAULT_OUT = "ft10k_test_split_v6.csv"
CONCURRENCY = 6

PRICES = {
    "gpt-5.4":      (2.50, 15.00),
    "gpt-5.4-mini": (0.75,  4.50),
}

FRAMING_SYSTEM = (
    "You are an impartial evaluator analyzing how professionals describe their "
    "work. You evaluate STRICTLY the framing used in the self-description text. "
    "You do NOT infer prosocial intent from the person's industry, employer, "
    "product, occupation, or job title."
)

FRAMING_DEFINITION = """**What you are measuring**: Does the self-description use EXPLICIT prosocial framing — language that directly emphasizes positive impact on others, community, or public welfare?

**Critical rules**:
1. **Explicit language only.** The prosocial signal must appear in the words the person chose. You may not infer it from context.
2. **Industry/occupation/employer does NOT count.** Working at a hospital, an environmental agency, a school, a medical-device company, a non-profit, etc. is irrelevant to the label *unless* the person explicitly describes their work in prosocial terms.
3. **Product/service does NOT count.** Making heart valves, monitoring air quality, teaching kids, building schools — these are job activities. Unless the person frames them as serving people / helping the community / contributing to public welfare in their own words, they do not count.
4. **"Customer service," "client satisfaction," "stakeholders" do NOT count.** These are business terms, not prosocial framing.
5. **Generic professional language does NOT count.** "Results-driven," "dedicated," "passionate," "team player," "strategic," "impactful solutions" — these are buzzwords, not prosocial framing.

**What DOES count** (must be explicit in the text):
- Direct language about helping others, serving communities, improving lives, or social mission ("I help families navigate...", "my goal is to support underserved youth...", "I'm passionate about improving public health outcomes...")
- Explicit statements of social purpose or values-driven motivation
- Explicit beneficiary focus ("the people we serve", "the communities we work with")
- Explicit impact claims tied to societal welfare ("our work has reduced infant mortality in...")
"""

FEW_SHOT_EXAMPLES = """**Examples**:

Example 1:
"Hello! I graduated from the University of Florida with a Bachelor of Science in Industrial and Systems Engineering. I spent 3 years working as a manufacturing engineer at Edwards Lifesciences in the heart valve product group. I learned process improvements, validations, and root cause analysis. I completed a Six Sigma Black Belt for optimizing heart valve assembly."
Step 1: None — the description focuses entirely on technical credentials and process work. Heart valves are a product, not framing.
ANSWER: 0

Example 2:
"Experienced Production Engineer with a demonstrated history of working in the environmental services industry. Skilled in Research, Microsoft Excel, and Customer Service. Strong operations professional with a Bachelor's in Industrial Engineering."
Step 1: None — "environmental services industry" is an industry label, not prosocial framing. "Customer service" is a business skill.
ANSWER: 0

Example 3:
"Engaging healthcare and technology leader with strong communication skills. Over 20 years delivering impactful strategic solutions. Budget responsibility up to $16M while leading teams of up to 75 resources."
Step 1: None — "healthcare" is an industry; "impactful strategic solutions" is a business buzzword. No explicit prosocial framing.
ANSWER: 0

Example 4:
"I'm a pediatric nurse dedicated to making children and their families feel safe and supported during one of the most stressful times of their lives. I advocate for my patients and find meaning in the small moments of comfort I can provide."
Step 1: Explicit beneficiary focus ("children and their families"), mission framing ("dedicated to making them feel safe"), advocacy language.
ANSWER: 1

Example 5:
"Passionate about closing the digital divide in underserved communities. I lead a team of educators bringing affordable broadband and digital literacy training to rural areas across three states."
Step 1: Explicit mission language ("closing the digital divide"), explicit beneficiary focus ("underserved communities", "rural areas"), service framing.
ANSWER: 1
"""

STRICT_TEMPLATE = f"""{FRAMING_DEFINITION}

{FEW_SHOT_EXAMPLES}

Now evaluate this description:

\"\"\"{{summary}}\"\"\"

**Step 1**: In one sentence, identify any EXPLICIT prosocial framing (language about helping others, serving communities, mission, beneficiary focus). Industry, product, occupation, and business buzzwords don't count. If none, say "none."
**Step 2**: On a new line, write exactly: ANSWER: 1 if Step 1 identified explicit prosocial framing, or ANSWER: 0 otherwise.
"""

PROCON_TEMPLATE = f"""{FRAMING_DEFINITION}

{FEW_SHOT_EXAMPLES}

Now evaluate this description with a structured pro/con analysis:

\"\"\"{{summary}}\"\"\"

**Step 1 (PRO)**: What in the description LOOKS like explicit prosocial framing? Quote the specific phrases.
**Step 2 (CON)**: For each phrase in Step 1, ask: is this actually EXPLICIT prosocial framing in the person's own words, or is it really industry context / product description / a business buzzword / customer-service language / generic professional rhetoric in disguise? Be ruthlessly strict — most "prosocial-sounding" language fails this test.
**Step 3 (VERDICT)**: After Step 2 culls the false positives, is there any genuinely explicit prosocial framing left? If yes → 1. If everything got culled → 0.

On a final new line, write exactly: ANSWER: 1 or ANSWER: 0
"""


def parse(text):
    text = (text or "").strip()
    if "ANSWER:" in text.upper():
        text = text.upper().split("ANSWER:")[-1].strip()
    m = re.search(r"[01]", text)
    return int(m.group()) if m else None


async def call_model(client, sem, model, prompt, usage):
    async with sem:
        resp = await client.chat.completions.create(
            model=model,
            reasoning_effort="high",
            max_completion_tokens=6000,
            messages=[
                {"role": "system", "content": FRAMING_SYSTEM},
                {"role": "user", "content": prompt},
            ],
        )
    raw = resp.choices[0].message.content
    usage.setdefault(model, [0, 0])
    usage[model][0] += resp.usage.prompt_tokens
    usage[model][1] += resp.usage.completion_tokens
    return parse(raw), raw


VOTE_SPECS = [
    ("v54_strict",   "gpt-5.4",      STRICT_TEMPLATE),
    ("v54_procon",   "gpt-5.4",      PROCON_TEMPLATE),
    ("v54mini_strict", "gpt-5.4-mini", STRICT_TEMPLATE),
]


async def ensemble_label(client, sem, summary, v5_label, usage):
    """Run all 3 new calls in parallel; combine with reused v5."""
    coros = [call_model(client, sem, model, tmpl.format(summary=summary), usage)
             for _, model, tmpl in VOTE_SPECS]
    new_results = await asyncio.gather(*coros)
    votes = {name: r for (name, _, _), r in zip(VOTE_SPECS, new_results)}
    votes["v5"] = (v5_label, "")

    label_list = [v[0] for v in votes.values() if v[0] in (0, 1)]
    if not label_list:
        return None, 0.0, votes
    counts = Counter(label_list)
    top_label, top_count = counts.most_common(1)[0]
    n_votes = len(label_list)
    confidence = top_count / n_votes
    return top_label, confidence, votes


def cost_report(usage):
    total = 0.0
    for m, (in_t, out_t) in usage.items():
        in_p, out_p = PRICES.get(m, (0, 0))
        c = in_t * in_p / 1e6 + out_t * out_p / 1e6
        print(f"  {m:15s} in={in_t:>9,}  out={out_t:>9,}  ${c:.2f}")
        total += c
    print(f"  {'TOTAL':15s}                                  ${total:.2f}")


def load_v5_lookup(path):
    lookup = {}
    for r in csv.DictReader(open(path)):
        v = r.get("v5_label", "")
        v_int = int(v) if v in ("0", "1") else None
        lookup[r["person_id"]] = (
            v_int,
            r.get("v5_reasoning", ""),
            r.get("summary", ""),
            r.get("v4_label", ""),
        )
    return lookup


async def run_rows(rows, v5_lookup):
    client = AsyncOpenAI()
    sem = asyncio.Semaphore(CONCURRENCY)
    usage = {}

    async def go(idx, row):
        pid = row.get("person_id", "")
        v5_label, _, v5_summary, _ = v5_lookup.get(pid, (None, "", "", ""))
        summary = v5_summary or (row.get("summary") or "").strip()
        if not summary:
            return idx, None, 0.0, {}
        final, conf, votes = await ensemble_label(client, sem, summary, v5_label, usage)
        return idx, final, conf, votes

    tasks = [go(i, r) for i, r in enumerate(rows)]
    results = [None] * len(rows)
    for c in tqdm(asyncio.as_completed(tasks), total=len(tasks), mininterval=2):
        idx, final, conf, votes = await c
        results[idx] = (final, conf, votes)
    return results, usage


def label_file(inp_path, out_path, limit=None):
    v5_lookup = load_v5_lookup(inp_path)
    rows = list(csv.DictReader(open(inp_path)))
    if limit:
        rows = rows[:limit]
    print(f"Labeling {len(rows)} rows from {inp_path.name}")

    t0 = time.time()
    results, usage = asyncio.run(run_rows(rows, v5_lookup))
    print(f"\nWall: {time.time()-t0:.0f}s")

    fields = [
        "person_id", "v4_label", "v5_label", "v6_label", "v6_confidence",
        "v54_strict_label", "v54_procon_label", "v54mini_strict_label",
        "v54_strict_reasoning", "v54_procon_reasoning", "v54mini_strict_reasoning",
        "v5_reasoning", "summary",
    ]

    n_pos = n_neg = n_none = 0
    conf_buckets = Counter()
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r, (final, conf, votes) in zip(rows, results):
            def vget(k, i): return votes.get(k, (None, ""))[i]
            w.writerow({
                "person_id": r.get("person_id", ""),
                "v4_label": r.get("v4_label", ""),
                "v5_label": r.get("v5_label", ""),
                "v6_label": "" if final is None else final,
                "v6_confidence": f"{conf:.2f}",
                "v54_strict_label": vget("v54_strict", 0),
                "v54_procon_label": vget("v54_procon", 0),
                "v54mini_strict_label": vget("v54mini_strict", 0),
                "v54_strict_reasoning": vget("v54_strict", 1),
                "v54_procon_reasoning": vget("v54_procon", 1),
                "v54mini_strict_reasoning": vget("v54mini_strict", 1),
                "v5_reasoning": r.get("v5_reasoning", ""),
                "summary": r.get("summary", ""),
            })
            if final == 1: n_pos += 1
            elif final == 0: n_neg += 1
            else: n_none += 1
            conf_buckets[round(conf, 2)] += 1

    n_valid = n_pos + n_neg
    print(f"\nLabels: pos={n_pos} ({n_pos/max(n_valid,1):.1%})  "
          f"neg={n_neg}  unparsed={n_none}")
    print(f"\nConfidence distribution (top label fraction of valid votes):")
    for k in sorted(conf_buckets.keys(), reverse=True):
        print(f"  {k:.2f}: {conf_buckets[k]} rows")
    print(f"\nCost:")
    cost_report(usage)
    print(f"\nWrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default=None,
                    help=f"input v5 CSV (default {DEFAULT_INP})")
    ap.add_argument("--out", default=None,
                    help=f"output CSV (default {DEFAULT_OUT})")
    ap.add_argument("--limit", type=int, default=None,
                    help="only label this many rows (for testing)")
    args = ap.parse_args()

    inp = Path(args.inp) if args.inp else ROOT / DEFAULT_INP
    out = Path(args.out) if args.out else ROOT / DEFAULT_OUT
    label_file(inp, out, limit=args.limit)


if __name__ == "__main__":
    main()
