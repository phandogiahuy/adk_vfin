from __future__ import annotations
    
from io import BytesIO
from typing import Callable, Dict, Iterable
from decimal import Decimal

import pandas as pd
from fastapi import UploadFile

from . import bq

_REQUIRED_TX_COLUMNS = {"store_id", "store_name", "region", "province", "date", "revenue", "cogs"}
_TX_NUMERIC_COLUMNS: Iterable[str] = [
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

_REQUIRED_AUDIT_COLUMNS = {"audit_id", "store_id", "planned_date"}
_REQUIRED_INCIDENT_COLUMNS = {"incident_id", "store_id", "opened_ts", "severity"}
_REQUIRED_RULE_COLUMNS = {"rule_id", "title"}


def _read_upload(upload: UploadFile) -> pd.DataFrame:
    filename = (upload.filename or "uploaded_file").lower()
    content = upload.file.read()
    if not content:
        raise ValueError("Uploaded file is empty.")
    buffer = BytesIO(content)
    try:
        if filename.endswith((".xlsx", ".xlsm", ".xls")):
            df = pd.read_excel(buffer)
        else:
            df = pd.read_csv(buffer)
    except Exception as exc:
        raise ValueError(f"Failed to read file: {exc}") from exc
    return df


def _normalize_transactions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    missing = _REQUIRED_TX_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().all():
        raise ValueError("Unable to parse 'date' column.")
    df["date"] = df["date"].dt.tz_localize(None)

    for col in _TX_NUMERIC_COLUMNS:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df["total_orders"] = df["total_orders"].astype(int)
    df["total_refunds"] = df["total_refunds"].astype(int)

    decimal_cols = [
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
    ]

    def _to_decimal(val):
        if pd.isna(val):
            return None
        return Decimal(str(val))

    for col in decimal_cols:
        df[col] = df[col].apply(_to_decimal)

    if "notes" not in df.columns:
        df["notes"] = ""
    if "created_ts" not in df.columns:
        df["created_ts"] = pd.Timestamp.utcnow()
    else:
        df["created_ts"] = pd.to_datetime(df["created_ts"], errors="coerce").fillna(pd.Timestamp.utcnow())
        df["created_ts"] = df["created_ts"].dt.tz_localize(None)

    return df


def _normalize_audit_schedule(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    missing = _REQUIRED_AUDIT_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    df["planned_date"] = pd.to_datetime(df["planned_date"], errors="coerce")
    if df["planned_date"].isna().all():
        raise ValueError("Unable to parse 'planned_date'.")
    df["planned_date"] = df["planned_date"].dt.tz_localize(None)

    if "actual_date" in df.columns:
        df["actual_date"] = pd.to_datetime(df["actual_date"], errors="coerce").dt.tz_localize(None)
    else:
        df["actual_date"] = pd.NaT

    optional_cols = ["region", "auditor", "scope", "status", "notes"]
    for col in optional_cols:
        if col not in df.columns:
            df[col] = ""

    df["scope"] = df["scope"].apply(lambda x: (x or "").strip())

    return df[
        ["audit_id", "store_id", "region", "planned_date", "actual_date", "auditor", "scope", "status", "notes"]
    ]


def _normalize_incidents(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    missing = _REQUIRED_INCIDENT_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    df["opened_ts"] = pd.to_datetime(df["opened_ts"], errors="coerce")
    if df["opened_ts"].isna().all():
        raise ValueError("Unable to parse 'opened_ts'.")
    df["opened_ts"] = df["opened_ts"].dt.tz_localize(None)

    if "closed_ts" in df.columns:
        df["closed_ts"] = pd.to_datetime(df["closed_ts"], errors="coerce").dt.tz_localize(None)
    else:
        df["closed_ts"] = pd.NaT

    optional_cols = ["region", "rule_id", "status", "owner", "summary", "corrective_action"]
    for col in optional_cols:
        if col not in df.columns:
            df[col] = ""

    return df[
        [
            "incident_id",
            "store_id",
            "region",
            "rule_id",
            "opened_ts",
            "closed_ts",
            "status",
            "severity",
            "owner",
            "summary",
            "corrective_action",
        ]
    ]


def _normalize_rules(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    missing = _REQUIRED_RULE_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    defaults = {
        "category": "",
        "description": "",
        "severity": "",
        "metric": "",
        "threshold": 0.0,
        "responsible_role": "",
        "remediation": "",
        "last_updated": pd.NaT,
    }
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default

    df["threshold"] = pd.to_numeric(df["threshold"], errors="coerce").fillna(0)
    df["last_updated"] = pd.to_datetime(df["last_updated"], errors="coerce").dt.tz_localize(None)

    return df[
        [
            "rule_id",
            "category",
            "title",
            "description",
            "severity",
            "metric",
            "threshold",
            "responsible_role",
            "remediation",
            "last_updated",
        ]
    ]


_DATASET_CONFIG: Dict[str, Dict[str, Callable[[pd.DataFrame], pd.DataFrame] | str]] = {
    "transactions": {
        "table": "store_daily_transactions",
        "normalizer": _normalize_transactions,
    },
    "store_daily_transactions": {
        "table": "store_daily_transactions",
        "normalizer": _normalize_transactions,
    },
    "audit_schedule": {
        "table": "audit_schedule",
        "normalizer": _normalize_audit_schedule,
    },
    "compliance_incidents": {
        "table": "compliance_incidents",
        "normalizer": _normalize_incidents,
    },
    "compliance_rules": {
        "table": "compliance_rules",
        "normalizer": _normalize_rules,
    },
}


async def ingest_dataset(upload: UploadFile, dataset_type: str = "transactions") -> dict:
    dataset_type = (dataset_type or "transactions").strip().lower()
    config = _DATASET_CONFIG.get(dataset_type)
    if not config:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

    upload.file.seek(0)
    df = _read_upload(upload)
    normalizer = config["normalizer"]
    df = normalizer(df)
    rows = bq.load_dataframe_to_table(df, config["table"])
    return {
        "rows": rows,
        "filename": upload.filename or "uploaded_file",
        "dataset_type": dataset_type,
        "table": config["table"],
    }
