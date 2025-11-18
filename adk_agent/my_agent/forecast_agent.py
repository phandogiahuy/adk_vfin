from __future__ import annotations

import os
from typing import Any, Dict

import pandas as pd
from dotenv import load_dotenv
from google.cloud import bigquery

from my_agent.macro import compute_macro_factor, get_macro_snapshot

load_dotenv()

PROJECT_ID = os.getenv("PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT")
DATASET = os.getenv("BQ_DATASET") or os.getenv("DATASET", "vfin_sme")
FORECAST_MODEL = os.getenv("FORECAST_MODEL", "sales_forecast_model")


def _model_path() -> str:
    if not PROJECT_ID:
        raise RuntimeError("PROJECT_ID is not configured")
    return f"{PROJECT_ID}.{DATASET}.{FORECAST_MODEL}"


def forecast_next_days(days: int = 7) -> Dict[str, Any]:
    if not PROJECT_ID:
        raise ValueError("PROJECT_ID is not configured; cannot query BigQuery model.")
    horizon = max(1, min(int(days), 90))
    client = bigquery.Client(project=PROJECT_ID)
    sql = f"""
        SELECT store_id, forecast_timestamp AS date, forecast_value AS revenue_forecast
        FROM ML.FORECAST(
            MODEL `{_model_path()}`,
            STRUCT({horizon} AS horizon)
        )
    """
    try:
        df = client.query(sql).result().to_dataframe()
        if df.empty:
            raise ValueError("BigQuery ML forecast returned 0 rows. Ensure the model is trained and has enough data.")
    except Exception as exc:
        raise RuntimeError(f"BigQuery forecast failed: {exc}") from exc

    macro_snapshot = get_macro_snapshot()
    macro_factor, contributions = compute_macro_factor(macro_snapshot)
    df["revenue_forecast"] = df["revenue_forecast"].astype(float)
    df["macro_adjusted_forecast"] = df["revenue_forecast"] * macro_factor

    return {
        "rows": df.to_dict(orient="records"),
        "macro": {
            "snapshot": macro_snapshot,
            "factor": macro_factor,
            "contributions": contributions,
        },
        "model": _model_path(),
    }
