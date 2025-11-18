from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

_PROJECT_ID = os.getenv("PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT")
_DATASET = os.getenv("BQ_DATASET") or os.getenv("DATASET", "vfin_sme")
_TABLE = os.getenv("BQ_TABLE")

_SAMPLE_CSV = os.path.join("vfin-sme", "sample", "vfin_sme_sample.csv")
_NUMERIC_COLUMNS: Iterable[str] = [
    "revenue",
    "cogs",
    "labor_cost",
    "rent_cost",
    "utilities_cost",
    "marketing_cost",
    "other_cost",
    "opening_cash",
    "closing_cash",
    "stock_value",
    "total_orders",
    "total_refunds",
]

_SCHEMA_CACHE: Dict[str, set[str]] = {}


def _has_bigquery() -> bool:
    return bool(_PROJECT_ID)


def _client():
    if not _has_bigquery():
        raise RuntimeError("PROJECT_ID not configured for BigQuery access.")
    from google.cloud import bigquery  # type: ignore

    return bigquery.Client(project=_PROJECT_ID)


def _table_fqn() -> str:
    if not _has_bigquery():
        raise RuntimeError("PROJECT_ID not configured for BigQuery access.")
    return f"{_PROJECT_ID}.{_DATASET}.{_TABLE}"


def _table_name(table: str) -> str:
    if not _has_bigquery():
        raise RuntimeError("PROJECT_ID not configured for BigQuery access.")
    return f"{_PROJECT_ID}.{_DATASET}.{table}"


def _table_fields(table_fqn: str) -> set[str]:
    if table_fqn in _SCHEMA_CACHE:
        return _SCHEMA_CACHE[table_fqn]
    from google.cloud import bigquery  # type: ignore

    client = _client()
    table = client.get_table(table_fqn)
    fields = {field.name for field in table.schema}
    _SCHEMA_CACHE[table_fqn] = fields
    return fields


def _run_query(sql: str, params: List[Any] | None = None) -> pd.DataFrame:
    if not _has_bigquery():
        return pd.DataFrame()
    from google.cloud import bigquery  # type: ignore

    client = _client()
    job_config = bigquery.QueryJobConfig(query_parameters=params or [])
    return client.query(sql, job_config=job_config).result().to_dataframe()


def load_dataframe(df: pd.DataFrame) -> int:
    """Append dataframe to default BigQuery table (or no-op locally)."""

    return load_dataframe_to_table(df, _TABLE)


def load_dataframe_to_table(df: pd.DataFrame, table: str) -> int:
    """Append dataframe to a specific BigQuery table."""

    if not _has_bigquery():
        return len(df)

    from google.cloud import bigquery  # type: ignore

    client = _client()
    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_APPEND")
    job = client.load_table_from_dataframe(df, _table_name(table), job_config=job_config)
    job.result()
    return job.output_rows or len(df)


def insert_contract_record(record: Dict[str, Any], table: str | None = None) -> None:
    """Insert a single contract record into BigQuery."""

    if not _has_bigquery():
        return

    from google.cloud import bigquery  # type: ignore

    client = _client()
    table_name = _table_name(table or os.getenv("CONTRACT_TABLE", "contract_intelligence"))

    # Ensure JSON-serialisable types (lists/dicts OK).
    table_name = _table_name(table or os.getenv("CONTRACT_TABLE", "contract_intelligence"))
    allowed_fields = _table_fields(table_name)

    # Only keep store_id, upload_date, contract_summary even if more fields exist.
    filtered: Dict[str, Any] = {}
    for key in ("store_id", "upload_date", "contract_summary"):
        if key in allowed_fields and key in record:
            value = record[key]
            if isinstance(value, pd.Timestamp):
                filtered[key] = value.isoformat()
            else:
                filtered[key] = value

    if not filtered:
        raise RuntimeError("No contract fields available to insert.")

    errors = client.insert_rows_json(table_name, [filtered])
    if errors:
        raise RuntimeError(f"Failed inserting contract record: {errors}")


def contracts_intelligence(year: int | None = None, month: int | None = None, limit: int = 200) -> pd.DataFrame:
    if not _has_bigquery():
        return pd.DataFrame()
    from google.cloud import bigquery  # type: ignore
    import time

    start = time.time()
    params: List[Any] = [bigquery.ScalarQueryParameter("limit", "INT64", limit)]
    filters: List[str] = []
    if year is not None:
        filters.append("EXTRACT(YEAR FROM DATE(upload_date)) = @year")
        params.append(bigquery.ScalarQueryParameter("year", "INT64", year))
    if month is not None:
        filters.append("EXTRACT(MONTH FROM DATE(upload_date)) = @month")
        params.append(bigquery.ScalarQueryParameter("month", "INT64", month))
    where_clause = " AND ".join(filters) if filters else "TRUE"
    sql = f"""
    SELECT
        store_id,
        upload_date,
        contract_summary
    FROM `{_table_name("contract_intelligence")}`
    WHERE {where_clause}
    ORDER BY upload_date DESC
    LIMIT @limit
    """
    df = _run_query(sql, params)
    print(f"[bq] contracts_intelligence fetched {len(df)} rows in {time.time()-start:.2f}s", flush=True)
    return df


def _load_sample_df() -> pd.DataFrame:
    if not os.path.exists(_SAMPLE_CSV):
        return pd.DataFrame([])
    df = pd.read_csv(_SAMPLE_CSV)
    df["date"] = pd.to_datetime(df["date"])
    return df


def _ensure_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for col in _NUMERIC_COLUMNS:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df["total_orders"] = df["total_orders"].astype(int)
    df["total_refunds"] = df["total_refunds"].astype(int)
    return df


def summary_by_region(
    days: int | None = None, year: int | None = None, month: int | None = None, day: int | None = None
) -> pd.DataFrame:
    if _has_bigquery():
        from google.cloud import bigquery  # type: ignore

        filters = []
        params = []
        if days is not None:
            filters.append("date >= DATE_SUB(CURRENT_DATE(), INTERVAL @days DAY)")
            params.append(bigquery.ScalarQueryParameter("days", "INT64", days))
        if year is not None:
            filters.append("EXTRACT(YEAR FROM date) = @year")
            params.append(bigquery.ScalarQueryParameter("year", "INT64", year))
        if month is not None:
            filters.append("EXTRACT(MONTH FROM date) = @month")
            params.append(bigquery.ScalarQueryParameter("month", "INT64", month))
        if day is not None:
            filters.append("EXTRACT(DAY FROM date) = @day")
            params.append(bigquery.ScalarQueryParameter("day", "INT64", day))
        where_clause = " AND ".join(filters) if filters else "TRUE"

        sql = f"""
        WITH base AS (
            SELECT
                region,
                CAST(date AS DATE) AS date,
                revenue,
                cogs,
                labor_cost,
                rent_cost,
                utilities_cost,
                marketing_cost,
                other_cost,
                total_orders,
                total_refunds
            FROM `{_table_fqn()}`
            WHERE {where_clause}
        )
        SELECT
            region,
            SUM(revenue) AS revenue,
            SUM(cogs) AS cogs,
            SUM(revenue - cogs) AS gross_profit,
            SUM(labor_cost + rent_cost + utilities_cost + marketing_cost + other_cost) AS opex,
            SUM((revenue - cogs) - (labor_cost + rent_cost + utilities_cost + marketing_cost + other_cost)) AS ebit,
            SUM(total_orders) AS total_orders,
            SUM(total_refunds) AS total_refunds
        FROM base
        GROUP BY region
        ORDER BY revenue DESC
        """
        client = _client()
        job_config = bigquery.QueryJobConfig(query_parameters=params)
        df = client.query(sql, job_config=job_config).result().to_dataframe()
        df["gross_margin"] = df["gross_profit"].where(df["revenue"] != 0, 0) / df["revenue"].where(
            df["revenue"] != 0, 1
        )
        return df

    # fallback
    df = _load_sample_df()
    if df.empty:
        return df
    recent = df
    if days is not None:
        recent = recent[recent["date"] >= (recent["date"].max() - pd.Timedelta(days=days))]
    if year is not None:
        recent = recent[recent["date"].dt.year == year]
    if month is not None:
        recent = recent[recent["date"].dt.month == month]
    if day is not None:
        recent = recent[recent["date"].dt.day == day]
    recent = _ensure_numeric(recent)
    grouped = (
        recent.groupby("region", as_index=False)[
            [
                "revenue",
                "cogs",
                "labor_cost",
                "rent_cost",
                "utilities_cost",
                "marketing_cost",
                "other_cost",
                "total_orders",
                "total_refunds",
            ]
        ]
        .sum()
    )
    grouped["gross_profit"] = grouped["revenue"] - grouped["cogs"]
    grouped["opex"] = (
        grouped["labor_cost"]
        + grouped["rent_cost"]
        + grouped["utilities_cost"]
        + grouped["marketing_cost"]
        + grouped["other_cost"]
    )
    grouped["ebit"] = grouped["gross_profit"] - grouped["opex"]
    grouped["gross_margin"] = grouped["gross_profit"].where(grouped["revenue"] != 0, 0) / grouped[
        "revenue"
    ].where(grouped["revenue"] != 0, 1)
    return grouped


def summary_by_store(
    days: int | None = None, year: int | None = None, month: int | None = None, day: int | None = None
) -> pd.DataFrame:
    if _has_bigquery():
        from google.cloud import bigquery  # type: ignore

        filters = []
        params = []
        if days is not None:
            filters.append("date >= DATE_SUB(CURRENT_DATE(), INTERVAL @days DAY)")
            params.append(bigquery.ScalarQueryParameter("days", "INT64", days))
        if year is not None:
            filters.append("EXTRACT(YEAR FROM date) = @year")
            params.append(bigquery.ScalarQueryParameter("year", "INT64", year))
        if month is not None:
            filters.append("EXTRACT(MONTH FROM date) = @month")
            params.append(bigquery.ScalarQueryParameter("month", "INT64", month))
        if day is not None:
            filters.append("EXTRACT(DAY FROM date) = @day")
            params.append(bigquery.ScalarQueryParameter("day", "INT64", day))
        where_clause = " AND ".join(filters) if filters else "TRUE"

        sql = f"""
        WITH base AS (
            SELECT
                store_id,
                ANY_VALUE(store_name) AS store_name,
                ANY_VALUE(region) AS region,
                CAST(date AS DATE) AS date,
                revenue,
                cogs,
                labor_cost,
                rent_cost,
                utilities_cost,
                marketing_cost,
                other_cost,
                total_orders,
                total_refunds
            FROM `{_table_fqn()}`
            WHERE {where_clause}
            GROUP BY store_id, date, revenue, cogs, labor_cost, rent_cost, utilities_cost, marketing_cost, other_cost, total_orders, total_refunds
        )
        SELECT
            store_id,
            ANY_VALUE(store_name) AS store_name,
            ANY_VALUE(region) AS region,
            SUM(revenue) AS revenue,
            SUM(cogs) AS cogs,
            SUM(revenue - cogs) AS gross_profit,
            SUM(labor_cost + rent_cost + utilities_cost + marketing_cost + other_cost) AS opex,
            SUM((revenue - cogs) - (labor_cost + rent_cost + utilities_cost + marketing_cost + other_cost)) AS ebit,
            SUM(total_orders) AS total_orders,
            SUM(total_refunds) AS total_refunds
        FROM base
        GROUP BY store_id
        ORDER BY revenue DESC
        """
        client = _client()
        job_config = bigquery.QueryJobConfig(query_parameters=params)
        df = client.query(sql, job_config=job_config).result().to_dataframe()
        if df.empty:
            return df
        df = _ensure_numeric(df)
        for col in [
            "revenue",
            "cogs",
            "gross_profit",
            "opex",
            "ebit",
            "labor_cost",
            "rent_cost",
            "utilities_cost",
            "marketing_cost",
            "other_cost",
        ]:
            if col in df.columns:
                df[col] = df[col].astype(float)
        df["gross_margin"] = df["gross_profit"].where(df["revenue"] != 0, 0) / df["revenue"].where(
            df["revenue"] != 0, 1
        )
        df["gross_margin"] = df["gross_margin"].astype(float)
        return df

    # fallback: reuse sample
    df = _load_sample_df()
    if df.empty:
        return df
    recent = df
    if days is not None:
        recent = recent[recent["date"] >= (recent["date"].max() - pd.Timedelta(days=days))]
    if year is not None:
        recent = recent[recent["date"].dt.year == year]
    if month is not None:
        recent = recent[recent["date"].dt.month == month]
    if day is not None:
        recent = recent[recent["date"].dt.day == day]
    recent = _ensure_numeric(recent)
    grouped = (
        recent.groupby(["store_id"], as_index=False)
        .agg(
            {
                "store_name": "first",
                "region": "first",
                "revenue": "sum",
                "cogs": "sum",
                "labor_cost": "sum",
                "rent_cost": "sum",
                "utilities_cost": "sum",
                "marketing_cost": "sum",
                "other_cost": "sum",
                "total_orders": "sum",
                "total_refunds": "sum",
            }
        )
    )
    grouped["gross_profit"] = grouped["revenue"] - grouped["cogs"]
    grouped["opex"] = (
        grouped["labor_cost"]
        + grouped["rent_cost"]
        + grouped["utilities_cost"]
        + grouped["marketing_cost"]
        + grouped["other_cost"]
    )
    grouped["ebit"] = grouped["gross_profit"] - grouped["opex"]
    grouped[[
        "revenue",
        "cogs",
        "gross_profit",
        "opex",
        "ebit",
        "labor_cost",
        "rent_cost",
        "utilities_cost",
        "marketing_cost",
        "other_cost",
    ]] = grouped[[
        "revenue",
        "cogs",
        "gross_profit",
        "opex",
        "ebit",
        "labor_cost",
        "rent_cost",
        "utilities_cost",
        "marketing_cost",
        "other_cost",
    ]].astype(float)
    grouped["gross_margin"] = grouped["gross_profit"].where(grouped["revenue"] != 0, 0) / grouped[
        "revenue"
    ].where(grouped["revenue"] != 0, 1)
    grouped["gross_margin"] = grouped["gross_margin"].astype(float)
    return grouped


def summary_records(
    days: int | None = None, year: int | None = None, month: int | None = None, day: int | None = None
) -> list[dict[str, Any]]:
    df = summary_by_region(days=days, year=year, month=month, day=day)
    if df.empty:
        return []
    df = df.fillna(0)
    return df.to_dict(orient="records")


def compliance_scores(days: int | None = None, year: int | None = None, month: int | None = None, limit: int = 50) -> pd.DataFrame:
    if not _has_bigquery():
        return pd.DataFrame()
    from google.cloud import bigquery  # type: ignore

    import time
    start = time.time()
    params: List[Any] = [bigquery.ScalarQueryParameter("limit", "INT64", limit)]
    filters: List[str] = []
    if days is not None:
        filters.append("l.audit_ts >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @days DAY)")
        params.append(bigquery.ScalarQueryParameter("days", "INT64", days))
    if year is not None:
        filters.append("EXTRACT(YEAR FROM l.audit_ts) = @score_year")
        params.append(bigquery.ScalarQueryParameter("score_year", "INT64", year))
    if month is not None:
        filters.append("EXTRACT(MONTH FROM l.audit_ts) = @score_month")
        params.append(bigquery.ScalarQueryParameter("score_month", "INT64", month))
    where_clause = " AND ".join(filters) if filters else "TRUE"

    sql = f"""
    SELECT
        l.store_id,
        s.store_name,
        s.region,
        s.province,
        l.rule_id,
        r.title AS rule_title,
        r.severity,
        r.threshold,
        l.score,
        l.status,
        l.audit_ts,
        l.auditor,
        l.notes
    FROM `{_table_name("store_compliance_log")}` AS l
    LEFT JOIN `{_table_name("compliance_rules")}` AS r USING (rule_id)
    LEFT JOIN `{_table_name("dim_store")}` AS s USING (store_id)
    WHERE {where_clause}
        ORDER BY l.audit_ts DESC
        LIMIT @limit
        """
    df = _run_query(sql, params)
    print(f"[bq] compliance_scores fetched {len(df)} rows in {time.time()-start:.2f}s", flush=True)
    return df


def compliance_incidents(
    open_only: bool = True, year: int | None = None, month: int | None = None, limit: int = 50
) -> pd.DataFrame:
    if not _has_bigquery():
        return pd.DataFrame()
    from google.cloud import bigquery  # type: ignore

    where: List[str] = []
    params: List[Any] = [bigquery.ScalarQueryParameter("limit", "INT64", limit)]
    if open_only:
        where.append("LOWER(i.status) != 'closed'")
    if year is not None:
        where.append("EXTRACT(YEAR FROM i.opened_ts) = @incident_year")
        params.append(bigquery.ScalarQueryParameter("incident_year", "INT64", year))
    if month is not None:
        where.append("EXTRACT(MONTH FROM i.opened_ts) = @incident_month")
        params.append(bigquery.ScalarQueryParameter("incident_month", "INT64", month))
    where_clause = " AND ".join(where) if where else "TRUE"

    sql = f"""
    SELECT
        i.incident_id,
        i.store_id,
        s.store_name,
        s.region,
        i.rule_id,
        r.title AS rule_title,
        i.opened_ts,
        i.closed_ts,
        i.status,
        i.severity,
        i.owner,
        i.summary,
        i.corrective_action
    FROM `{_table_name("compliance_incidents")}` AS i
    LEFT JOIN `{_table_name("compliance_rules")}` AS r USING (rule_id)
    LEFT JOIN `{_table_name("dim_store")}` AS s USING (store_id)
    WHERE {where_clause}
    ORDER BY i.opened_ts DESC
    LIMIT @limit
    """
    return _run_query(sql, params)


def upcoming_audits(year: int | None = None, month: int | None = None, limit: int = 20) -> pd.DataFrame:
    if not _has_bigquery():
        return pd.DataFrame()
    from google.cloud import bigquery  # type: ignore

    params: List[Any] = [bigquery.ScalarQueryParameter("limit", "INT64", limit)]
    filters: List[str] = ["IFNULL(LOWER(a.status), 'planned') != 'completed'"]
    if year is not None:
        filters.append("EXTRACT(YEAR FROM a.planned_date) = @audit_year")
        params.append(bigquery.ScalarQueryParameter("audit_year", "INT64", year))
    if month is not None:
        filters.append("EXTRACT(MONTH FROM a.planned_date) = @audit_month")
        params.append(bigquery.ScalarQueryParameter("audit_month", "INT64", month))
    where_clause = " AND ".join(filters)

    sql = f"""
    SELECT
        a.audit_id,
        a.store_id,
        s.store_name,
        s.region,
        a.planned_date,
        a.actual_date,
        a.auditor,
        a.scope,
        a.status,
        a.notes
    FROM `{_table_name("audit_schedule")}` AS a
    LEFT JOIN `{_table_name("dim_store")}` AS s USING (store_id)
    WHERE {where_clause}
    ORDER BY a.planned_date
    LIMIT @limit
    """
    return _run_query(sql, params)


def checklist_failures(
    days: int | None = 30, year: int | None = None, month: int | None = None, limit: int = 50
) -> pd.DataFrame:
    if not _has_bigquery():
        return pd.DataFrame()
    from google.cloud import bigquery  # type: ignore

    params: List[Any] = [
        bigquery.ScalarQueryParameter("limit", "INT64", limit),
    ]
    where = ["LOWER(result) = 'fail'"]
    if days is not None:
        where.append("captured_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @days DAY)")
        params.append(bigquery.ScalarQueryParameter("days", "INT64", days))
    if year is not None:
        where.append("EXTRACT(YEAR FROM captured_at) = @chk_year")
        params.append(bigquery.ScalarQueryParameter("chk_year", "INT64", year))
    if month is not None:
        where.append("EXTRACT(MONTH FROM captured_at) = @chk_month")
        params.append(bigquery.ScalarQueryParameter("chk_month", "INT64", month))
    where_clause = " AND ".join(where)

    sql = f"""
    SELECT
        r.response_id,
        r.store_id,
        s.store_name,
        s.region,
        r.checklist_name,
        r.item,
        r.comment,
        r.captured_at,
        r.media_url
    FROM `{_table_name("store_checklist_responses")}` AS r
    LEFT JOIN `{_table_name("dim_store")}` AS s USING (store_id)
    WHERE {where_clause}
    ORDER BY r.captured_at DESC
    LIMIT @limit
    """
    return _run_query(sql, params)


def store_financial_snapshot(days: int = 30, limit: int = 50) -> pd.DataFrame:
    if _has_bigquery():
        from google.cloud import bigquery  # type: ignore

        table_name = _table_name("store_daily_transactions")
        params: List[Any] = [
            bigquery.ScalarQueryParameter("days", "INT64", days),
            bigquery.ScalarQueryParameter("limit", "INT64", limit),
        ]
        sql = f"""
        WITH base AS (
            SELECT
                store_id,
                store_name,
                region,
                revenue,
                cogs,
                (labor_cost + rent_cost + utilities_cost + marketing_cost + other_cost) AS opex
            FROM `{table_name}`
            WHERE date >= DATE_SUB(CURRENT_DATE(), INTERVAL @days DAY)
        )
        SELECT
            store_id,
            ANY_VALUE(store_name) AS store_name,
            ANY_VALUE(region) AS region,
            SUM(revenue) AS revenue,
            SUM(cogs) AS cogs,
            SUM(opex) AS opex,
            SAFE_DIVIDE(SUM(revenue - cogs - opex), NULLIF(SUM(revenue), 0)) AS ebit_margin,
            SAFE_DIVIDE(SUM(revenue - cogs), NULLIF(SUM(revenue), 0)) AS gross_margin
        FROM base
        GROUP BY store_id
        ORDER BY revenue DESC
        LIMIT @limit
        """
        return _run_query(sql, params)

    df = _load_sample_df()
    if df.empty:
        return df
    recent = df[df["date"] >= (df["date"].max() - pd.Timedelta(days=days))]
    recent = _ensure_numeric(recent)
    grouped = recent.groupby(["store_id", "store_name", "region"], as_index=False).agg(
        {
            "revenue": "sum",
            "cogs": "sum",
            "labor_cost": "sum",
            "rent_cost": "sum",
            "utilities_cost": "sum",
            "marketing_cost": "sum",
            "other_cost": "sum",
        }
    )
    grouped["opex"] = (
        grouped["labor_cost"]
        + grouped["rent_cost"]
        + grouped["utilities_cost"]
        + grouped["marketing_cost"]
        + grouped["other_cost"]
    )
    grouped["ebit_margin"] = (
        grouped["revenue"] - grouped["cogs"] - grouped["opex"]
    ).where(grouped["revenue"] != 0, 0) / grouped["revenue"].where(grouped["revenue"] != 0, 1)
    grouped["gross_margin"] = (
        grouped["revenue"] - grouped["cogs"]
    ).where(grouped["revenue"] != 0, 0) / grouped["revenue"].where(grouped["revenue"] != 0, 1)
    cols = ["store_id", "store_name", "region", "revenue", "cogs", "opex", "ebit_margin", "gross_margin"]
    return grouped[cols].head(limit)
