"""Run composition-adjusted robustness checks for the findings report."""

from pathlib import Path
import os
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
    last = None
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
            last = error
            if attempt < 5:
                time.sleep(2)
    raise last


def fetch(conn, name, sql):
    saved = OUT / f"{name}.csv"
    if os.environ.get("RESUME") == "1" and saved.exists():
        frame = pd.read_csv(saved)
        print(f"Reusing {name}: {len(frame):,} rows", flush=True)
        return frame
    print(f"Running {name}...", flush=True)
    cur = conn.cursor()
    try:
        cur.execute(sql, timeout=420)
        frame = pd.DataFrame(cur.fetchall(), columns=[c[0] for c in cur.description])
    finally:
        cur.close()
    frame.to_csv(OUT / f"{name}.csv", index=False)
    print(f"  {len(frame):,} rows", flush=True)
    return frame


def contrast(frame, category, left, right):
    share = frame.pivot(index="OCCUPATION", columns=category, values="POSITIVE_SHARE")
    sizes = frame.pivot(index="OCCUPATION", columns=category, values="N")
    keep = share[[left, right]].dropna().index
    gaps = (100 * (share.loc[keep, left] - share.loc[keep, right])).astype(float)
    weights = sizes.loc[keep, [left, right]].min(axis=1).astype(float)
    return {
        "comparison": f"{left} minus {right}",
        "occupations": len(gaps),
        "positive_occupations": int((gaps > 0).sum()),
        "median_gap_pp": float(gaps.median()),
        "weighted_gap_pp": float(np.average(gaps, weights=weights)),
        "p10_gap_pp": float(gaps.quantile(0.10)),
        "p90_gap_pp": float(gaps.quantile(0.90)),
    }


def main():
    conn = None if os.environ.get("RESUME") == "1" else connect()
    try:
        org = fetch(
            conn,
            "company_type_by_occupation",
            """
            SELECT JOB_ONET_BROAD_OCCUPATION AS OCCUPATION,
                   LOWER(JOB_COMPANY_TYPE) AS COMPANY_TYPE,
                   COUNT(*) AS N, AVG(LABEL) AS POSITIVE_SHARE
            FROM PDL_CLEAN.BGI_2026_05.FRAMING_ANALYSIS_PERSON
            WHERE JOB_ONET_BROAD_OCCUPATION IS NOT NULL
              AND LOWER(JOB_COMPANY_TYPE) IN
                  ('nonprofit','educational','government','private','public','public_subsidiary')
            GROUP BY 1, 2 HAVING COUNT(*) >= 2000
            """,
        )
        size = fetch(
            conn,
            "company_size_by_occupation",
            """
            SELECT JOB_ONET_BROAD_OCCUPATION AS OCCUPATION,
                   CASE WHEN JOB_COMPANY_SIZE IN ('1-10','11-50') THEN 'small_1_50'
                        WHEN JOB_COMPANY_SIZE IN ('5001-10000','10001+') THEN 'large_5001_plus'
                   END AS SIZE_BAND,
                   COUNT(*) AS N, AVG(LABEL) AS POSITIVE_SHARE
            FROM PDL_CLEAN.BGI_2026_05.FRAMING_ANALYSIS_PERSON
            WHERE JOB_ONET_BROAD_OCCUPATION IS NOT NULL
              AND JOB_COMPANY_SIZE IN ('1-10','11-50','5001-10000','10001+')
            GROUP BY 1, 2 HAVING COUNT(*) >= 2000
            """,
        )
        cohort = fetch(
            conn,
            "birth_cohort_by_occupation",
            """
            SELECT JOB_ONET_BROAD_OCCUPATION AS OCCUPATION,
                   CASE WHEN BIRTH_YEAR BETWEEN 1980 AND 1999 THEN 'born_1980_1999'
                        WHEN BIRTH_YEAR BETWEEN 1950 AND 1969 THEN 'born_1950_1969'
                   END AS COHORT,
                   COUNT(*) AS N, AVG(LABEL) AS POSITIVE_SHARE
            FROM PDL_CLEAN.BGI_2026_05.FRAMING_ANALYSIS_PERSON
            WHERE JOB_ONET_BROAD_OCCUPATION IS NOT NULL
              AND BIRTH_YEAR BETWEEN 1950 AND 1999
              AND NOT (BIRTH_YEAR BETWEEN 1970 AND 1979)
            GROUP BY 1, 2 HAVING COUNT(*) >= 2000
            """,
        )
        summary_len = fetch(
            conn,
            "summary_length",
            """
            SELECT CASE WHEN p.SUMMARY IS NULL OR LENGTH(TRIM(p.SUMMARY)) = 0 THEN 'Missing/empty'
                        WHEN LENGTH(p.SUMMARY) < 100 THEN '<100 chars'
                        WHEN LENGTH(p.SUMMARY) < 250 THEN '100-249 chars'
                        WHEN LENGTH(p.SUMMARY) < 500 THEN '250-499 chars'
                        WHEN LENGTH(p.SUMMARY) < 1000 THEN '500-999 chars'
                        ELSE '1000+ chars' END AS SUMMARY_LENGTH,
                   COUNT(*) AS N, AVG(s.ENSEMBLE_PROB) AS AVG_PROB,
                   AVG(s.LABEL) AS POSITIVE_SHARE
            FROM PDL_CLEAN.BGI_2026_05.FRAMING_SCORES_V6_SEED123 s
            INNER JOIN PDL_CLEAN.BGI_2026_05.ROOT_PERSON p
              ON s.PERSON_ID = p.PERSON_ID
            GROUP BY 1 ORDER BY MIN(COALESCE(LENGTH(p.SUMMARY), 0))
            """,
        )
        prob_bins = fetch(
            conn,
            "probability_bins",
            """
            SELECT CASE WHEN ENSEMBLE_PROB < 0.001 THEN '<0.001'
                        WHEN ENSEMBLE_PROB < 0.01 THEN '0.001-0.01'
                        WHEN ENSEMBLE_PROB < 0.10 THEN '0.01-0.10'
                        WHEN ENSEMBLE_PROB < 0.50 THEN '0.10-0.50'
                        WHEN ENSEMBLE_PROB < 0.80 THEN '0.50-0.80'
                        WHEN ENSEMBLE_PROB < 0.99 THEN '0.80-0.99'
                        ELSE '0.99-1.00' END AS PROBABILITY_BIN,
                   COUNT(*) AS N
            FROM PDL_CLEAN.BGI_2026_05.FRAMING_SCORES_V6_SEED123
            GROUP BY 1 ORDER BY MIN(ENSEMBLE_PROB)
            """,
        )
        major_sex = fetch(
            conn,
            "major_family_by_sex",
            """
            WITH person_major AS (
              SELECT DISTINCT e.PERSON_ID, LOWER(p.SEX) AS SEX,
                     e.ENSEMBLE_PROB, e.LABEL,
                     LEFT(REGEXP_REPLACE(TO_VARCHAR(e.BGI_MAJOR_CIP6_CODE), '[^0-9]', ''), 2) AS CIP2
              FROM PDL_CLEAN.BGI_2026_05.FRAMING_ANALYSIS_EDUCATION e
              INNER JOIN PDL_CLEAN.BGI_2026_05.FRAMING_ANALYSIS_PERSON p
                ON e.PERSON_ID = p.PERSON_ID
              WHERE LOWER(p.SEX) IN ('female','male')
                AND e.BGI_MAJOR_CIP6_CODE IS NOT NULL
            )
            SELECT CASE
                     WHEN CIP2 IN ('05','09','16','23','24','30','38','42','44','45','50','54') THEN 'Liberal arts / social sciences / arts'
                     WHEN CIP2 IN ('11','14','15','26','27','40','41') THEN 'STEM'
                     WHEN CIP2 = '51' THEN 'Health'
                     WHEN CIP2 = '52' THEN 'Business'
                     WHEN CIP2 = '13' THEN 'Education'
                     ELSE 'Other fields' END AS MAJOR_FAMILY,
                   SEX, COUNT(DISTINCT PERSON_ID) AS N,
                   AVG(LABEL) AS POSITIVE_SHARE
            FROM person_major WHERE LENGTH(CIP2) = 2
            GROUP BY 1, 2 HAVING COUNT(DISTINCT PERSON_ID) >= 10000
            ORDER BY 1, 2
            """,
        )
    finally:
        if conn is not None:
            conn.close()

    checks = [
        contrast(org, "COMPANY_TYPE", "nonprofit", "private"),
        contrast(org, "COMPANY_TYPE", "educational", "private"),
        contrast(org, "COMPANY_TYPE", "government", "private"),
        contrast(org, "COMPANY_TYPE", "private", "public"),
        contrast(size, "SIZE_BAND", "small_1_50", "large_5001_plus"),
        contrast(cohort, "COHORT", "born_1980_1999", "born_1950_1969"),
    ]
    checks = pd.DataFrame(checks)
    checks.to_csv(OUT / "within_occupation_robustness.csv", index=False)

    print("\nWITHIN-OCCUPATION ROBUSTNESS")
    print(checks.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print("\nSUMMARY LENGTH")
    print(summary_len.to_string(index=False))
    print("\nPROBABILITY BINS")
    print(prob_bins.to_string(index=False))
    print("\nMAJOR FAMILY BY SEX")
    print(major_sex.to_string(index=False))


if __name__ == "__main__":
    main()
