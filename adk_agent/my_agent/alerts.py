from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

PROJECT_ID = os.getenv("PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT")
DATASET = os.getenv("BQ_DATASET") or os.getenv("DATASET", "vfin_sme")
TABLE = os.getenv("ALERTS_TABLE", "alerts_log")


def _table_fqn() -> str:
    if not PROJECT_ID:
        raise RuntimeError("PROJECT_ID not configured")
    return f"{PROJECT_ID}.{DATASET}.{TABLE}"


def log_alert(region: Optional[str], message: str, alert_type: str = "nudge", confidence: float = 0.8) -> None:
    """Persist alert record into BigQuery if PROJECT_ID is available."""

    if not PROJECT_ID:
        return
    try:
        from google.cloud import bigquery  # type: ignore

        client = bigquery.Client(project=PROJECT_ID)
        rows = [
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "region": region or "-",
                "message": message,
                "type": alert_type,
                "confidence": float(confidence),
            }
        ]
        client.insert_rows_json(_table_fqn(), rows)
    except Exception:
        # Swallow logging errors so they don't break user flow
        return

