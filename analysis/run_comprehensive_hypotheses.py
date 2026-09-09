"""Run a broad, reproducible hypothesis sweep over the production framing scores."""

from pathlib import Path
import os
import re
import time

import numpy as np
import pandas as pd
from dotenv import load_dotenv
import snowflake.connector


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
load_dotenv(ROOT / ".env")


def connect():
    last_error = None
    for attempt in range(1, 6):
        try:
            print(f"Snowflake connection attempt {attempt}/5", flush=True)
            return snowflake.connector.connect(
                user=os.environ["SNOWFLAKE_USER"],
                password=os.environ["SNOWFLAKE_PASSWORD"],
                account=os.environ["SNOWFLAKE_ACCOUNT"],
                warehouse=os.environ.get("SNOWFLAKE_WAREHOUSE", "WH_4_XS"),
                role=os.environ.get("SNOWFLAKE_ROLE"),
                login_timeout=30,
                network_timeout=180,
            )
        except Exception as error:
            last_error = error
            if attempt < 5:
                time.sleep(2)
    raise last_error


def fetch(conn, name, sql):
    output_path = OUT / f"{name}.csv"
    if os.environ.get("RESUME") == "1" and output_path.exists():
        frame = pd.read_csv(output_path)
        print(f"Reusing {name}: {len(frame):,} rows", flush=True)
        return frame
    print(f"Running {name}...", flush=True)
    cur = conn.cursor()
    try:
        cur.execute(sql, timeout=300)
        rows = cur.fetchall()
        columns = [column[0] for column in cur.description]
    finally:
        cur.close()
    frame = pd.DataFrame(rows, columns=columns)
    frame.to_csv(output_path, index=False)
    print(f"  {len(frame):,} rows", flush=True)
    return frame


def normalize_name(value):
    if pd.isna(value):
        return ""
    value = str(value).lower().replace("&", " and ")
    value = re.sub(r"^the\s+", "", value)
    value = re.sub(r"\b(incorporated|inc)\.?$", "", value)
    return re.sub(r"[^a-z0-9]+", "", value)


def weighted_mean(group, value="POSITIVE_SHARE", weight="N"):
    return float(np.average(group[value].astype(float), weights=group[weight].astype(float)))


def weighted_groups(frame, fields):
    rows = []
    for keys, group in frame.groupby(fields, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(fields, keys))
        row.update(
            institutions=len(group),
            N=int(group["N"].sum()),
            AVG_PROB=weighted_mean(group, "AVG_PROB"),
            POSITIVE_SHARE=weighted_mean(group),
            COLLEGE_WEIGHTED_SHARE=float(group["POSITIVE_SHARE"].astype(float).mean()),
        )
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    conn = connect()
    try:
        results = {}
        results["overall"] = fetch(
            conn,
            "overall",
            "SELECT * FROM PDL_CLEAN.BGI_2026_05.FRAMING_ANALYSIS_OVERALL",
        )
        results["occupation_broad"] = fetch(
            conn,
            "occupation_broad",
            "SELECT * FROM PDL_CLEAN.BGI_2026_05.FRAMING_ANALYSIS_OCCUPATION_BROAD ORDER BY POSITIVE_SHARE DESC",
        )
        results["occupation_specific"] = fetch(
            conn,
            "occupation_specific",
            "SELECT * FROM PDL_CLEAN.BGI_2026_05.FRAMING_ANALYSIS_OCCUPATION ORDER BY POSITIVE_SHARE DESC",
        )
        results["industry"] = fetch(
            conn,
            "industry",
            "SELECT * FROM PDL_CLEAN.BGI_2026_05.FRAMING_ANALYSIS_INDUSTRY ORDER BY POSITIVE_SHARE DESC",
        )
        results["company_type"] = fetch(
            conn,
            "company_type",
            "SELECT * FROM PDL_CLEAN.BGI_2026_05.FRAMING_ANALYSIS_COMPANY_TYPE ORDER BY POSITIVE_SHARE DESC",
        )
        results["company_size"] = fetch(
            conn,
            "company_size",
            """
            SELECT JOB_COMPANY_SIZE AS GROUP_NAME, COUNT(*) AS N,
                   AVG(ENSEMBLE_PROB) AS AVG_PROB, AVG(LABEL) AS POSITIVE_SHARE
            FROM PDL_CLEAN.BGI_2026_05.FRAMING_ANALYSIS_PERSON
            WHERE JOB_COMPANY_SIZE IS NOT NULL
            GROUP BY 1 HAVING COUNT(*) >= 1000
            ORDER BY POSITIVE_SHARE DESC
            """,
        )
        results["birth_decade"] = fetch(
            conn,
            "birth_decade",
            """
            SELECT FLOOR(BIRTH_YEAR / 10) * 10 AS BIRTH_DECADE, COUNT(*) AS N,
                   AVG(ENSEMBLE_PROB) AS AVG_PROB, AVG(LABEL) AS POSITIVE_SHARE
            FROM PDL_CLEAN.BGI_2026_05.FRAMING_ANALYSIS_PERSON
            WHERE BIRTH_YEAR BETWEEN 1940 AND 2005
            GROUP BY 1 HAVING COUNT(*) >= 10000
            ORDER BY 1
            """,
        )
        results["job_role"] = fetch(
            conn,
            "job_role",
            """
            SELECT JOB_TITLE_ROLE AS GROUP_NAME, COUNT(*) AS N,
                   AVG(ENSEMBLE_PROB) AS AVG_PROB, AVG(LABEL) AS POSITIVE_SHARE
            FROM PDL_CLEAN.BGI_2026_05.FRAMING_ANALYSIS_PERSON
            WHERE JOB_TITLE_ROLE IS NOT NULL
            GROUP BY 1 HAVING COUNT(*) >= 10000
            ORDER BY POSITIVE_SHARE DESC
            """,
        )
        results["state"] = fetch(
            conn,
            "state",
            "SELECT * FROM PDL_CLEAN.BGI_2026_05.FRAMING_ANALYSIS_STATE ORDER BY POSITIVE_SHARE DESC",
        )
        results["metro"] = fetch(
            conn,
            "metro",
            "SELECT * FROM PDL_CLEAN.BGI_2026_05.FRAMING_ANALYSIS_METRO ORDER BY POSITIVE_SHARE DESC",
        )
        results["sex_by_occupation"] = fetch(
            conn,
            "sex_by_occupation",
            """
            SELECT JOB_ONET_BROAD_OCCUPATION AS OCCUPATION, LOWER(SEX) AS SEX,
                   COUNT(*) AS N, AVG(ENSEMBLE_PROB) AS AVG_PROB,
                   AVG(LABEL) AS POSITIVE_SHARE
            FROM PDL_CLEAN.BGI_2026_05.FRAMING_ANALYSIS_PERSON
            WHERE JOB_ONET_BROAD_OCCUPATION IS NOT NULL
              AND LOWER(SEX) IN ('female', 'male')
            GROUP BY 1, 2 HAVING COUNT(*) >= 5000
            """,
        )
        results["race_by_occupation"] = fetch(
            conn,
            "race_by_occupation",
            """
            SELECT JOB_ONET_BROAD_OCCUPATION AS OCCUPATION, BGI_RACE AS RACE,
                   COUNT(*) AS N, AVG(ENSEMBLE_PROB) AS AVG_PROB,
                   AVG(LABEL) AS POSITIVE_SHARE
            FROM PDL_CLEAN.BGI_2026_05.FRAMING_ANALYSIS_PERSON
            WHERE JOB_ONET_BROAD_OCCUPATION IS NOT NULL AND BGI_RACE IS NOT NULL
            GROUP BY 1, 2 HAVING COUNT(*) >= 5000
            """,
        )
        results["degree_level"] = fetch(
            conn,
            "degree_level",
            """
            WITH ranked AS (
              SELECT PERSON_ID, ANY_VALUE(ENSEMBLE_PROB) AS ENSEMBLE_PROB,
                     ANY_VALUE(LABEL) AS LABEL,
                     MAX(CASE
                           WHEN BGI_DEGREE ILIKE '%doctor%' THEN 5
                           WHEN BGI_DEGREE ILIKE '%master%' THEN 4
                           WHEN BGI_DEGREE ILIKE '%bachelor%' THEN 3
                           WHEN BGI_DEGREE ILIKE '%associate%' THEN 2
                           WHEN BGI_DEGREE ILIKE '%certificate%' THEN 1
                           ELSE 0 END) AS DEGREE_RANK
              FROM PDL_CLEAN.BGI_2026_05.FRAMING_ANALYSIS_EDUCATION
              GROUP BY PERSON_ID
            )
            SELECT DEGREE_RANK,
                   CASE DEGREE_RANK WHEN 5 THEN 'Doctorate' WHEN 4 THEN 'Master''s'
                     WHEN 3 THEN 'Bachelor''s' WHEN 2 THEN 'Associate'
                     WHEN 1 THEN 'Certificate' ELSE 'Other/unspecified' END AS GROUP_NAME,
                   COUNT(*) AS N, AVG(ENSEMBLE_PROB) AS AVG_PROB,
                   AVG(LABEL) AS POSITIVE_SHARE
            FROM ranked GROUP BY 1, 2 ORDER BY DEGREE_RANK DESC
            """,
        )
        results["major_family"] = fetch(
            conn,
            "major_family",
            """
            WITH person_major AS (
              SELECT DISTINCT PERSON_ID, ENSEMBLE_PROB, LABEL,
                     LEFT(REGEXP_REPLACE(TO_VARCHAR(BGI_MAJOR_CIP6_CODE), '[^0-9]', ''), 2) AS CIP2
              FROM PDL_CLEAN.BGI_2026_05.FRAMING_ANALYSIS_EDUCATION
              WHERE BGI_MAJOR_CIP6_CODE IS NOT NULL
            )
            SELECT CASE
                     WHEN CIP2 IN ('05','09','16','23','24','30','38','42','44','45','50','54') THEN 'Liberal arts / social sciences / arts'
                     WHEN CIP2 IN ('11','14','15','26','27','40','41') THEN 'STEM'
                     WHEN CIP2 = '51' THEN 'Health'
                     WHEN CIP2 = '52' THEN 'Business'
                     WHEN CIP2 = '13' THEN 'Education'
                     WHEN CIP2 IN ('01','03') THEN 'Agriculture / natural resources'
                     WHEN CIP2 IN ('46','47','48','49') THEN 'Trades / transportation'
                     ELSE 'Other fields' END AS GROUP_NAME,
                   COUNT(DISTINCT PERSON_ID) AS N,
                   AVG(ENSEMBLE_PROB) AS AVG_PROB, AVG(LABEL) AS POSITIVE_SHARE
            FROM person_major WHERE LENGTH(CIP2) = 2
            GROUP BY 1 HAVING COUNT(DISTINCT PERSON_ID) >= 10000
            ORDER BY POSITIVE_SHARE DESC
            """,
        )
        schools = fetch(
            conn,
            "college_school_results",
            "SELECT GROUP_NAME, N, AVG_PROB, POSITIVE_SHARE FROM PDL_CLEAN.BGI_2026_05.FRAMING_ANALYSIS_COLLEGE",
        )
    finally:
        conn.close()

    # Within-occupation gender gaps.
    sex = results["sex_by_occupation"]
    p_share = sex.pivot(index="OCCUPATION", columns="SEX", values="POSITIVE_SHARE").dropna()
    p_n = sex.pivot(index="OCCUPATION", columns="SEX", values="N").reindex(p_share.index)
    gender = pd.DataFrame(
        {
            "OCCUPATION": p_share.index,
            "FEMALE_SHARE": p_share["female"].values,
            "MALE_SHARE": p_share["male"].values,
            "GAP_PP": 100 * (p_share["female"] - p_share["male"]).values,
            "MIN_GROUP_N": p_n[["female", "male"]].min(axis=1).values,
        }
    ).sort_values("GAP_PP", ascending=False)
    gender.to_csv(OUT / "gender_gap_within_occupation.csv", index=False)

    # Institution attributes from the local IPEDS/Carnegie crosswalk.
    ipeds = pd.read_csv(ROOT / "data" / "ipeds_master.csv", low_memory=False)
    schools["name_key"] = schools["GROUP_NAME"].map(normalize_name)
    ipeds["name_key"] = ipeds["INSTNM"].map(normalize_name)
    key_counts = ipeds.groupby("name_key")["UNITID"].transform("nunique")
    ipeds_unique = ipeds[key_counts == 1].drop_duplicates("name_key")
    college = schools.merge(
        ipeds_unique[
            [
                "name_key", "INSTNM", "C21BASIC", "carnegie_grp", "selectivity_label",
                "is_4_year", "is_public", "is_private_nfp", "is_private_fp", "is_hbcu_flag",
            ]
        ],
        on="name_key",
        how="inner",
        validate="many_to_one",
    )
    college = college[college["is_4_year"] == 1].copy()
    college["college_control"] = np.select(
        [college["is_public"] == 1, college["is_private_nfp"] == 1, college["is_private_fp"] == 1],
        ["Public", "Private nonprofit", "Private for-profit"],
        default="Other",
    )
    college["liberal_arts"] = np.where(college["C21BASIC"] == 21, "Liberal arts", "Other four-year")
    weighted_groups(college, ["carnegie_grp"]).sort_values("POSITIVE_SHARE", ascending=False).to_csv(
        OUT / "college_by_carnegie.csv", index=False
    )
    weighted_groups(college, ["college_control"]).sort_values("POSITIVE_SHARE", ascending=False).to_csv(
        OUT / "college_by_control.csv", index=False
    )
    weighted_groups(college, ["selectivity_label"]).sort_values("POSITIVE_SHARE", ascending=False).to_csv(
        OUT / "college_by_selectivity.csv", index=False
    )
    weighted_groups(college, ["liberal_arts", "selectivity_label"]).to_csv(
        OUT / "liberal_arts_by_selectivity.csv", index=False
    )
    weighted_groups(college, ["is_hbcu_flag"]).sort_values("POSITIVE_SHARE", ascending=False).to_csv(
        OUT / "college_hbcu.csv", index=False
    )

    print("\nKEY ROBUSTNESS SUMMARIES", flush=True)
    print(
        f"Gender gap: female share exceeds male share in "
        f"{(gender.GAP_PP > 0).sum()}/{len(gender)} matched broad occupations; "
        f"median gap {gender.GAP_PP.median():+.2f} pp."
    )
    print("\nMajor families")
    print(results["major_family"].to_string(index=False))
    print("\nDegree levels")
    print(results["degree_level"].to_string(index=False))
    print("\nCompany types")
    print(results["company_type"].to_string(index=False))
    print("\nCompany sizes")
    print(results["company_size"].to_string(index=False))
    print("\nBirth decades")
    print(results["birth_decade"].to_string(index=False))
    print("\nCollege control")
    print(pd.read_csv(OUT / "college_by_control.csv").to_string(index=False))
    print("\nCollege selectivity")
    print(pd.read_csv(OUT / "college_by_selectivity.csv").to_string(index=False))
    print("\nHBCU")
    print(pd.read_csv(OUT / "college_hbcu.csv").to_string(index=False))
    print("\nLargest within-occupation gender gaps")
    print(gender.head(15).to_string(index=False))
    print("\nSmallest/reversed within-occupation gender gaps")
    print(gender.tail(15).to_string(index=False))


if __name__ == "__main__":
    main()
