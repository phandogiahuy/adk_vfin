from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

PROJECT_ID = os.getenv("PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT")
DATASET = os.getenv("BQ_DATASET") or os.getenv("DATASET", "vfin_sme")


def _has_bigquery() -> bool:
    return bool(PROJECT_ID)


def _client():
    if not _has_bigquery():
        raise RuntimeError("PROJECT_ID not configured for BigQuery")
    from google.cloud import bigquery  # type: ignore

    return bigquery.Client(project=PROJECT_ID)


def _table(name: str) -> str:
    if not _has_bigquery():
        raise RuntimeError("PROJECT_ID not configured for BigQuery")
    return f"{PROJECT_ID}.{DATASET}.{name}"


def _query(sql: str, params: List[Any] | None = None) -> pd.DataFrame:
    if not _has_bigquery():
        return pd.DataFrame([])
    from google.cloud import bigquery  # type: ignore

    client = _client()
    job_config = None
    if params:
        job_config = bigquery.QueryJobConfig(query_parameters=params)
    return client.query(sql, job_config=job_config).result().to_dataframe()


def _build_time_filters(field: str, days: int | None, year: int | None, month: int | None, day: int | None):
    from google.cloud import bigquery  # type: ignore

    clauses = []
    params: List[Any] = []
    if days is not None:
        clauses.append(f"{field} >= DATE_SUB(CURRENT_DATE(), INTERVAL @days DAY)")
        params.append(bigquery.ScalarQueryParameter("days", "INT64", days))
    if year is not None:
        clauses.append(f"EXTRACT(YEAR FROM {field}) = @year")
        params.append(bigquery.ScalarQueryParameter("year", "INT64", year))
    if month is not None:
        clauses.append(f"EXTRACT(MONTH FROM {field}) = @month")
        params.append(bigquery.ScalarQueryParameter("month", "INT64", month))
    if day is not None:
        clauses.append(f"EXTRACT(DAY FROM {field}) = @day")
        params.append(bigquery.ScalarQueryParameter("day", "INT64", day))
    return (" AND ".join(clauses) if clauses else "TRUE", params)


def top_stores(
    days: int | None = None,
    year: int | None = None,
    month: int | None = None,
    day: int | None = None,
    limit: int = 3,
) -> List[Dict[str, Any]]:
    from google.cloud import bigquery  # type: ignore

    where_clause, params = _build_time_filters("date", days, year, month, day)
    sql = f"""
    WITH agg AS (
        SELECT
            store_id,
            ANY_VALUE(store_name) AS store_name,
            ANY_VALUE(region) AS region,
            SUM(revenue) AS revenue,
            SUM(cogs) AS cogs,
            SUM(revenue - cogs) AS gross_profit,
            SUM((revenue - cogs) - (labor_cost + rent_cost + utilities_cost + marketing_cost + other_cost)) AS ebit
        FROM `{_table("store_daily_transactions")}`
        WHERE {where_clause}
        GROUP BY store_id
    )
    SELECT * FROM agg
    ORDER BY ebit DESC
    LIMIT @limit
    """
    params = params + [bigquery.ScalarQueryParameter("limit", "INT64", limit)]
    df = _query(sql, params=params)
    return df.fillna(0).to_dict(orient="records") if not df.empty else []


def bottom_stores(
    days: int | None = None,
    year: int | None = None,
    month: int | None = None,
    day: int | None = None,
    limit: int = 3,
) -> List[Dict[str, Any]]:
    from google.cloud import bigquery  # type: ignore

    where_clause, params = _build_time_filters("date", days, year, month, day)
    sql = f"""
    WITH agg AS (
        SELECT
            store_id,
            ANY_VALUE(store_name) AS store_name,
            ANY_VALUE(region) AS region,
            SUM(revenue) AS revenue,
            SUM(cogs) AS cogs,
            SUM((revenue - cogs) - (labor_cost + rent_cost + utilities_cost + marketing_cost + other_cost)) AS ebit
        FROM `{_table("store_daily_transactions")}`
        WHERE {where_clause}
        GROUP BY store_id
    )
    SELECT * FROM agg
    ORDER BY ebit ASC
    LIMIT @limit
    """
    params = params + [bigquery.ScalarQueryParameter("limit", "INT64", limit)]
    df = _query(sql, params=params)
    return df.fillna(0).to_dict(orient="records") if not df.empty else []


def cash_health(days: int | None = None, year: int | None = None, month: int | None = None, day: int | None = None) -> List[Dict[str, Any]]:
    from google.cloud import bigquery  # type: ignore

    where_clause, params = _build_time_filters("date", days, year, month, day)
    sql = f"""
    SELECT
        region,
        SUM(opening_cash) AS opening_cash,
        SUM(closing_cash) AS closing_cash,
        SUM(closing_cash - opening_cash) AS cash_delta,
        SUM(stock_value) AS stock_value
    FROM `{_table("store_daily_transactions")}`
    WHERE {where_clause}
    GROUP BY region
    ORDER BY cash_delta
    """
    df = _query(sql, params=params)
    return df.fillna(0).to_dict(orient="records") if not df.empty else []


def recent_alerts(limit: int = 5, year: int | None = None, month: int | None = None, day: int | None = None) -> List[Dict[str, Any]]:
    from google.cloud import bigquery  # type: ignore

    where_clause, params = _build_time_filters("ts", None, year, month, day)
    sql = f"""
    SELECT ts, region, message, type, confidence
    FROM `{_table("alerts_log")}`
    WHERE {where_clause}
    ORDER BY ts DESC
    LIMIT @limit
    """
    params = params + [bigquery.ScalarQueryParameter("limit", "INT64", limit)]
    df = _query(sql, params=params)
    return df.to_dict(orient="records") if not df.empty else []


def forecast_snapshot(horizon: int = 14, limit: int = 50) -> List[Dict[str, Any]]:
    from google.cloud import bigquery  # type: ignore

    sql = f"""
    SELECT store_id, forecast_date, revenue_forecast, created_ts
    FROM `{_table("forecast_cache")}`
    WHERE forecast_date <= DATE_ADD(CURRENT_DATE(), INTERVAL @horizon DAY)
    ORDER BY forecast_date ASC, created_ts DESC
    LIMIT @limit
    """
    df = _query(
        sql,
        params=[
            bigquery.ScalarQueryParameter("horizon", "INT64", horizon),
            bigquery.ScalarQueryParameter("limit", "INT64", limit),
        ],
    )
    return df.to_dict(orient="records") if not df.empty else []


def report_history(limit: int = 5) -> List[Dict[str, Any]]:
    from google.cloud import bigquery  # type: ignore

    sql = f"""
    SELECT report_id, gcs_url, created_at
    FROM `{_table("reports_index")}`
    ORDER BY created_at DESC
    LIMIT @limit
    """
    df = _query(
        sql,
        params=[bigquery.ScalarQueryParameter("limit", "INT64", limit)],
    )
    return df.to_dict(orient="records") if not df.empty else []
