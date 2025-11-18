from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict, List, Optional

import pandas as pd

from my_agent import bq
from my_agent.alerts import log_alert


def _sanitize_value(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, list):
        return [_sanitize_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _sanitize_value(v) for k, v in value.items()}
    return value


def _pick_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for name in candidates:
        if name in df.columns:
            return name
    return None


def _store_label(row: Dict[str, Any]) -> str:
    return row.get("store_name") or row.get("store_id") or "Store"


def generate_nudges(days: int = 30) -> dict:
    df = bq.summary_by_store(days).fillna(0)
    if df.empty:
        return {"summary": "No recent store data found.", "nudges": []}

    revenue_col = _pick_column(df, ["revenue", "net_revenue"])
    margin_col = _pick_column(df, ["ebit_margin", "gross_margin", "margin"])
    growth_col = _pick_column(df, ["revenue_change", "delta_revenue", "sales_velocity"])
    stock_col = _pick_column(df, ["inventory_days", "stock_cover"])

    summary_bits: List[str] = [
        "Reviewed store performance over the most recent month." if days >= 28 else f"Assessed store performance over the last {days} day(s)."
    ]
    if revenue_col:
        total_rev = float(df[revenue_col].sum())
        summary_bits.append(f"Aggregate revenue ≈ {total_rev:,.0f}.")
    if margin_col:
        avg_margin = float(df[margin_col].mean())
        summary_bits.append(f"Average margin around {avg_margin:.1f}%.")
    summary_text = " ".join(summary_bits)

    nudges: List[Dict[str, Any]] = []

    if margin_col:
        for row in df.nsmallest(min(2, len(df)), margin_col).to_dict(orient="records"):
            margin = row.get(margin_col)
            nudges.append(
                {
                    "title": f"Protect {_store_label(row)} margin",
                    "action": "Deploy targeted price or cost-action plan.",
                    "reason": f"{_store_label(row)} margin at {margin:.1f}% trailing {days} days.",
                    "timeframe": "Today",
                }
            )

    if growth_col:
        for row in df.nlargest(min(2, len(df)), growth_col).to_dict(orient="records"):
            growth = row.get(growth_col)
            nudges.append(
                {
                    "title": f"Double down on {_store_label(row)} momentum",
                    "action": "Secure inventory and staffing to sustain the uplift.",
                    "reason": f"{_store_label(row)} growth signal at {growth:.1f} on {growth_col}.",
                    "timeframe": "Next 7 days",
                }
            )

    if stock_col:
        for row in df.nlargest(min(1, len(df)), stock_col).to_dict(orient="records"):
            stock = row.get(stock_col)
            nudges.append(
                {
                    "title": f"Reduce stock overhang at {_store_label(row)}",
                    "action": "Launch markdown or transfer plan to normalize stock cover.",
                    "reason": f"{stock_col.replace('_', ' ').title()} at {stock:.1f} days.",
                    "timeframe": "Next 7 days",
                }
            )

    if not nudges:
        top_row = df.iloc[0].to_dict()
        nudges.append(
            {
                "title": f"Monitor {_store_label(top_row)} run-rate",
                "action": "Review pricing and staffing to keep momentum intact.",
                "reason": "General review surfaced no acute risks; maintain discipline.",
                "timeframe": "Next 7 days",
            }
        )

    log_alert(region=None, message="CEO nudges generated", alert_type="nudge", confidence=0.6)
    return {"summary": summary_text, "nudges": _sanitize_value(nudges)}
