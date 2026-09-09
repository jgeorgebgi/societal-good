"""Publish aggregate CSV results as simple Snowflake tables."""

from __future__ import annotations

import argparse
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import snowflake.connector
from dotenv import load_dotenv
from snowflake.connector.pandas_tools import write_pandas


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
load_dotenv(ROOT / ".env")


def identifier(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").upper()
    if not value or value[0].isdigit():
        value = f"DATA_{value}"
    return value


def connect(database: str, schema: str):
    return snowflake.connector.connect(
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        user=os.environ["SNOWFLAKE_USER"],
        password=os.environ["SNOWFLAKE_PASSWORD"],
        warehouse=os.environ.get("SNOWFLAKE_WAREHOUSE", "WH_4_XS"),
        role=os.environ.get("SNOWFLAKE_ROLE") or None,
        database=database,
        schema=schema,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", default=os.environ.get("SNOWFLAKE_DATABASE", "PDL_CLEAN"))
    parser.add_argument("--schema", default=os.environ.get("SNOWFLAKE_SCHEMA", "BGI_2026_05"))
    parser.add_argument("--prefix", default="SOCIETAL_GOOD_")
    parser.add_argument("--replace", action="store_true", help="replace existing aggregate tables")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    files = sorted(RESULTS.glob("*.csv"))
    if not files:
        raise SystemExit(f"No CSV files found in {RESULTS}")

    plan = [(path, identifier(args.prefix + path.stem)) for path in files]
    for path, table in plan:
        print(f"{path.name} -> {args.database}.{args.schema}.{table}")
    if args.dry_run:
        return

    connection = connect(args.database, args.schema)
    manifest = []
    try:
        for path, table in plan:
            frame = pd.read_csv(path)
            frame.columns = [identifier(column) for column in frame.columns]
            success, _, rows, _ = write_pandas(
                connection,
                frame,
                table,
                database=args.database,
                schema=args.schema,
                auto_create_table=True,
                overwrite=args.replace,
                quote_identifiers=False,
            )
            if not success or rows != len(frame):
                raise RuntimeError(f"Incomplete upload for {path.name}: {rows}/{len(frame)} rows")
            manifest.append(
                {
                    "DATASET_NAME": path.stem,
                    "TABLE_NAME": table,
                    "ROW_COUNT": len(frame),
                    "LOADED_AT_UTC": datetime.now(timezone.utc).replace(tzinfo=None),
                }
            )

        manifest_frame = pd.DataFrame(manifest)
        success, _, rows, _ = write_pandas(
            connection,
            manifest_frame,
            identifier(args.prefix + "DATASETS"),
            database=args.database,
            schema=args.schema,
            auto_create_table=True,
            overwrite=True,
            quote_identifiers=False,
        )
        if not success or rows != len(manifest_frame):
            raise RuntimeError("Manifest upload was incomplete")
    finally:
        connection.close()

    print(f"Published {len(manifest)} datasets.")


if __name__ == "__main__":
    main()
