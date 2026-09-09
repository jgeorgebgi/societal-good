"""Test whether Carnegie liberal-arts colleges have higher framing scores."""

import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
import snowflake.connector


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


def normalize_name(value):
    if pd.isna(value):
        return ""
    value = str(value).lower().replace("&", " and ")
    value = re.sub(r"^the\s+", "", value)
    value = re.sub(r"\b(incorporated|inc)\.?$", "", value)
    return re.sub(r"[^a-z0-9]+", "", value)


def weighted_mean(frame, value, weight):
    return np.average(frame[value], weights=frame[weight])


def summarize(frame, label):
    rows = []
    for is_lac, group in frame.groupby("is_liberal_arts"):
        rows.append(
            {
                "sample": label,
                "group": "Liberal arts" if is_lac else "Other four-year",
                "institutions": len(group),
                "person_school_records": int(group["N"].sum()),
                "person_weighted_probability": weighted_mean(group, "AVG_PROB", "N"),
                "person_weighted_positive_share": weighted_mean(group, "POSITIVE_SHARE", "N"),
                "college_weighted_positive_share": group["POSITIVE_SHARE"].mean(),
            }
        )
    return pd.DataFrame(rows)


def main():
    conn = None
    last_error = None
    for attempt in range(5):
        try:
            conn = snowflake.connector.connect(
                user=os.environ["SNOWFLAKE_USER"],
                password=os.environ["SNOWFLAKE_PASSWORD"],
                account=os.environ["SNOWFLAKE_ACCOUNT"],
                warehouse=os.environ.get("SNOWFLAKE_WAREHOUSE", "WH_4_XS"),
                role=os.environ.get("SNOWFLAKE_ROLE"),
                login_timeout=30,
                network_timeout=120,
            )
            break
        except Exception as error:
            last_error = error
            if attempt < 4:
                time.sleep(2)
    if conn is None:
        raise last_error
    try:
        schools = pd.read_sql(
            """
            SELECT GROUP_NAME, N, AVG_PROB, POSITIVE_SHARE
            FROM PDL_CLEAN.BGI_2026_05.FRAMING_ANALYSIS_COLLEGE
            """,
            conn,
        )
    finally:
        conn.close()

    ipeds = pd.read_csv(ROOT / "data" / "ipeds_master.csv", low_memory=False)
    schools["name_key"] = schools["GROUP_NAME"].map(normalize_name)
    ipeds["name_key"] = ipeds["INSTNM"].map(normalize_name)
    # Without a school-state field in the aggregate table, same-name campuses
    # cannot be disambiguated reliably. Keep only names that map to one UNITID.
    key_counts = ipeds.groupby("name_key")["UNITID"].transform("nunique")
    ipeds = ipeds[key_counts == 1].sort_values(["name_key", "UNITID"]).drop_duplicates("name_key")

    merged = schools.merge(
        ipeds[
            [
                "name_key",
                "INSTNM",
                "C21BASIC",
                "carnegie_grp",
                "selectivity_label",
                "is_4_year",
                "is_public",
                "is_private_nfp",
            ]
        ],
        on="name_key",
        how="left",
        validate="many_to_one",
    )
    matched = merged[merged["C21BASIC"].notna()].copy()
    four_year = matched[matched["is_4_year"] == 1].copy()
    four_year["is_liberal_arts"] = four_year["C21BASIC"].eq(21)

    print(f"Snowflake institutions (N>=500): {len(schools):,}")
    print(f"Unique exact normalized-name IPEDS matches: {len(matched):,} ({len(matched)/len(schools):.1%})")
    print(f"Matched four-year institutions: {len(four_year):,}")

    outputs = [summarize(four_year, "N>=500")]
    outputs.append(summarize(four_year[four_year["N"] >= 2_000], "N>=2,000"))
    summary = pd.concat(outputs, ignore_index=True)
    print("\nRAW COMPARISON")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nDIFFERENCES (liberal arts minus other four-year)")
    for sample, part in summary.groupby("sample"):
        values = part.set_index("group")
        if {"Liberal arts", "Other four-year"}.issubset(values.index):
            person_diff = 100 * (
                values.loc["Liberal arts", "person_weighted_positive_share"]
                - values.loc["Other four-year", "person_weighted_positive_share"]
            )
            college_diff = 100 * (
                values.loc["Liberal arts", "college_weighted_positive_share"]
                - values.loc["Other four-year", "college_weighted_positive_share"]
            )
            print(f"{sample}: person-weighted {person_diff:+.2f} pp; college-weighted {college_diff:+.2f} pp")

    print("\nBY SELECTIVITY")
    stratified = (
        four_year.dropna(subset=["selectivity_label"])
        .groupby(["selectivity_label", "is_liberal_arts"], observed=True)
        .apply(
            lambda g: pd.Series(
                {
                    "institutions": len(g),
                    "records": int(g["N"].sum()),
                    "positive_share": weighted_mean(g, "POSITIVE_SHARE", "N"),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    stratified["group"] = np.where(stratified["is_liberal_arts"], "Liberal arts", "Other four-year")
    print(stratified.drop(columns="is_liberal_arts").to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nLARGEST MATCHED LIBERAL-ARTS COLLEGES")
    print(
        four_year[four_year["is_liberal_arts"]]
        .sort_values("N", ascending=False)
        [["GROUP_NAME", "N", "AVG_PROB", "POSITIVE_SHARE", "selectivity_label"]]
        .head(25)
        .to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )


if __name__ == "__main__":
    main()
