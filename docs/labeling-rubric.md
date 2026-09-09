# Gold hand-labeling rubric — prosocial framing (STRICT)

Label `your_label` in **gold_handlabel_v6.csv** with `0` or `1`. Use the `notes` column
for anything you're unsure about (helps us audit later). **Label blind** — don't look up the
v6 label while deciding.

## The question
**Does the self-description use EXPLICIT prosocial framing — language, in the person's own
words, that directly emphasizes positive impact on other people, communities, or public welfare?**

`1` = yes, explicit prosocial framing.
`0` = everything else.

## Counts as 1 (must be explicit in the text)
- Helping/serving people or communities: "I help families navigate…", "support underserved youth"
- Explicit social mission / values-driven purpose: "closing the digital divide"
- Named beneficiary focus: "the communities we serve", "at-risk students", "a deprived community"
- Impact tied to public welfare: "reduced infant mortality in…"

## Counts as 0 (does NOT qualify)
- Industry / employer / occupation (hospital, nonprofit, environmental agency) — context, not framing
- Product / service described as a job activity (makes heart valves, teaches kids) without prosocial framing
- **Customers / clients / stakeholders** — business terms, even "client wellbeing"
- Generic buzzwords: results-driven, passionate, dedicated, impactful, team player, strategic

## Edge-case rules (resolve the cases that split the LLMs)
1. **Client/customer wellbeing** → **0**. Serving paying clients isn't public welfare, even when warm.
2. **Mentoring/developing coworkers or "growing the company"** → **0**. Internal professional growth, not public benefit.
3. **"Patient-focused care" (generic)** → **0**. But a *specific* beneficiary/mission ("deprived community", "advocate for my patients") → **1**.
4. **Employer's mission stated as if one's own** ("at a company dedicated to healthcare access") → **0**. Must be the person's own framing.
5. **Generic inspire / role-model / "help others succeed"** → **0** unless a specific beneficiary beyond the workplace is named.
6. **⚠️ NAMING A CAUSE — YOU MUST DECIDE THIS ONE UP FRONT and apply it consistently:**
   Does "Passionate about sustainability, diversity, and clean energy" (a list of causes, no action)
   count as prosocial framing?
   - If **YES** (lenient) → stating commitment to a cause = `1`
   - If **NO** (strict) → a topic list without described action/beneficiary = `0`
   **Pick one before you start and never switch mid-set.** This single choice drives a big share of
   the disagreements, so your consistency on it is what makes the gold set trustworthy.

## Worked examples
- "Senior Technical Leader… scalable solutions for business success… shared values with principled people." → **0** (business)
- "Pediatric nurse dedicated to making children and their families feel safe during a stressful time." → **1** (beneficiary + mission)
- "World class customer service to assist customers, friends and family with their final-expense needs." → **0** (rule 1)
- "Inclusive teacher working with at-risk elementary students." → **1** (named beneficiary)
- "I help nonprofits find funding so they can grow their impact." → **decide with rule 1**: nonprofits are her *clients* → lean **0**.
