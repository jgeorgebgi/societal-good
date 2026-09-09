"""Build compact meeting charts from the saved aggregate analysis outputs."""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results"
OUT = DATA / "charts"
OUT.mkdir(exist_ok=True)

NAVY = "#15324A"
BLUE = "#2B6F9F"
TEAL = "#2A9D8F"
GOLD = "#E9A23B"
GRAY = "#B7C2CC"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titleweight": "bold"})


def finish(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def percent_axis(ax):
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.grid(axis="x", alpha=.18)
    ax.set_axisbelow(True)


major = pd.read_csv(DATA / "major_family.csv").sort_values("POSITIVE_SHARE")
fig, ax = plt.subplots(figsize=(8.5, 4.8))
colors = [TEAL if "Liberal" in x else BLUE for x in major.GROUP_NAME]
ax.barh(major.GROUP_NAME, major.POSITIVE_SHARE, color=colors)
percent_axis(ax)
ax.set_title("Positive-class rate by field of study")
ax.set_xlabel("Positive-class rate")
for i, v in enumerate(major.POSITIVE_SHARE): ax.text(v + .004, i, f"{v:.1%}", va="center")
ax.set_xlim(0, .42)
finish(fig, "major_families.png")

ctype = pd.read_csv(DATA / "company_type.csv").sort_values("POSITIVE_SHARE")
csize = pd.read_csv(DATA / "company_size.csv")
order = ["1-10", "11-50", "51-200", "201-500", "501-1000", "1001-5000", "5001-10000", "10001+"]
csize["GROUP_NAME"] = pd.Categorical(csize.GROUP_NAME, order, ordered=True)
csize = csize.sort_values("GROUP_NAME")
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
axes[0].barh(ctype.GROUP_NAME.str.replace("_", " "), ctype.POSITIVE_SHARE, color=BLUE)
axes[0].set_title("By employer type")
axes[1].plot(csize.POSITIVE_SHARE, csize.GROUP_NAME.astype(str), marker="o", color=TEAL, linewidth=2.5)
axes[1].set_title("By employer size")
for ax in axes: percent_axis(ax); ax.set_xlabel("Positive-class rate")
finish(fig, "employers.png")

robust = pd.read_csv(DATA / "within_occupation_robustness.csv").sort_values("weighted_gap_pp")
fig, ax = plt.subplots(figsize=(9, 4.8))
labels = (robust.comparison.str.replace(" minus ", " − ")
          .str.replace("born_1980_1999", "born 1980–1999")
          .str.replace("born_1950_1969", "born 1950–1969")
          .str.replace("small_1_50", "small employer")
          .str.replace("large_5001_plus", "large employer"))
ax.barh(labels, robust.weighted_gap_pp / 100, color=[GOLD if x < 0 else TEAL for x in robust.weighted_gap_pp])
percent_axis(ax)
ax.set_title("Differences that remain within occupation")
ax.set_xlabel("Weighted positive-class rate difference")
for i, v in enumerate(robust.weighted_gap_pp): ax.text(v/100 + .002, i, f"{v:+.1f} pp", va="center")
finish(fig, "within_occupation.png")

college = pd.read_csv(DATA / "liberal_arts_by_selectivity.csv")
p = college.pivot(index="selectivity_label", columns="liberal_arts", values="POSITIVE_SHARE")
p["gap"] = p["Liberal arts"] - p["Other four-year"]
p = p.sort_values("gap")
fig, ax = plt.subplots(figsize=(9, 4.8))
ax.barh(p.index, p.gap, color=TEAL)
percent_axis(ax)
ax.set_title("Liberal-arts college advantage within selectivity bands")
ax.set_xlabel("Positive-class rate difference")
for i, v in enumerate(p.gap): ax.text(v + .001, i, f"{v:+.1%}", va="center")
finish(fig, "liberal_arts_selectivity.png")

summary = pd.read_csv(DATA / "summary_length.csv")
fig, ax = plt.subplots(figsize=(8.5, 4.5))
ax.plot(summary.SUMMARY_LENGTH, summary.POSITIVE_SHARE, marker="o", linewidth=2.5, color=GOLD)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.set_ylim(0, .43)
ax.grid(axis="y", alpha=.18)
ax.set_title("Model output rises sharply with profile-summary length")
ax.set_ylabel("Positive-class rate")
ax.set_xlabel("Summary length")
for i, v in enumerate(summary.POSITIVE_SHARE): ax.text(i, v + .015, f"{v:.1%}", ha="center")
finish(fig, "summary_length.png")

print(f"Wrote charts to {OUT}")
