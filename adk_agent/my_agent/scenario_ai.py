from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from my_agent import bq

MAX_ROWS_FOR_ANALYSIS = 50


def _sanitize_record(value: Any) -> Any:
    if isinstance(value, list):
        return [_sanitize_record(v) for v in value]
    if isinstance(value, dict):
        return {k: _sanitize_record(v) for k, v in value.items()}
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, (pd.Series, pd.DataFrame)):
        return value.to_dict()
    if isinstance(value, (pd.Int64Dtype, pd.Float64Dtype)):
        return float(value)  # type: ignore[arg-type]
    return value


def _store_label(row: Dict[str, Any]) -> str:
    if row.get("store_name"):
        return str(row["store_name"])
    if row.get("store_id"):
        return str(row["store_id"])
    return "Store"


def _top_bottom_segments(df: pd.DataFrame, column: str, count: int = 3) -> List[Dict[str, Any]]:
    segments: List[Dict[str, Any]] = []
    if column not in df.columns:
        return segments
    for row in df.nlargest(count, column).to_dict(orient="records"):
        segments.append(
            {
                "segment": f"Leader — {_store_label(row)}",
                "stores": row.get("store_id"),
                "signal": f"{column} at {row[column]:.1f}",
                "implication": "Use as blueprint to lift lagging stores.",
            }
        )
    for row in df.nsmallest(count, column).to_dict(orient="records"):
        segments.append(
            {
                "segment": f"At risk — {_store_label(row)}",
                "stores": row.get("store_id"),
                "signal": f"{column} at {row[column]:.1f}",
                "implication": "Focus coaching, cost guardrails, and promo intensity.",
            }
        )
    return segments


def _scenario_plays(df: pd.DataFrame, target_margin: Optional[float]) -> List[Dict[str, str]]:
    plays: List[Dict[str, str]] = []
    leaders = df.nlargest(1, "revenue") if "revenue" in df.columns else df.head(1)
    laggards = df.nsmallest(1, "ebit_margin") if "ebit_margin" in df.columns else df.tail(1)
    if not leaders.empty:
        row = leaders.iloc[0].to_dict()
        plays.append(
            {
                "target": "Revenue momentum",
                "driver": _store_label(row),
                "adjustment": "Expand product depth + peak staffing.",
                "impact": "Sustain comp growth and pricing power.",
                "risk": "Inventory strain if demand cools suddenly.",
            }
        )
    if not laggards.empty:
        row = laggards.iloc[0].to_dict()
        plays.append(
            {
                "target": "Margin recovery",
                "driver": _store_label(row),
                "adjustment": "Tighten discounting + reschedule overtime.",
                "impact": "Adds 50–80 bps EBIT if executed in 4 weeks.",
                "risk": "Sales softness if price-value gap widens.",
            }
        )
    if target_margin is not None:
        plays.append(
            {
                "target": f"EBIT {target_margin:.1f}%",
                "driver": "Mix shift to high-margin regions",
                "adjustment": "Rebalance inventory and media spend to leaders.",
                "impact": "Accelerates blended margin uplift.",
                "risk": "Under-serving growth stores if reallocation is slow.",
            }
        )
    return plays


def _thirty_day_plan(goal: Optional[str]) -> List[Dict[str, str]]:
    plan: List[Dict[str, str]] = [
        {
            "owner": "Ops",
            "action": "Daily huddles on conversion and staffing by region.",
            "timing": "Week 1",
            "dependency": "Updated labor model",
        },
        {
            "owner": "Merch/Marketing",
            "action": "Push high-margin bundles in top 3 districts.",
            "timing": "Weeks 2-3",
            "dependency": "Inventory + creative approvals",
        },
        {
            "owner": "Finance",
            "action": "Margin monitoring cockpit with alerts.",
            "timing": "Week 4",
            "dependency": "Data engineering capacity",
        },
    ]
    if goal:
        plan.append(
            {
                "owner": "Strategy",
                "action": f"Translate goal '{goal}' into store targets.",
                "timing": "Week 1",
                "dependency": "Leadership sign-off",
            }
        )
    return plan


def ai_scenario_advise(days: int = 30, target_margin: float | None = None, goal: str | None = None) -> dict:
    snapshot = bq.store_financial_snapshot(days=days, limit=MAX_ROWS_FOR_ANALYSIS)
    if snapshot.empty:
        return {
            "timeframe": f"last {days} days",
            "summary": "No store data available for scenario planning.",
            "segments": [],
            "plays": [],
            "plan": [],
            "goal": goal,
            "target_margin": target_margin,
            "stores": [],
        }

    df = snapshot.copy()
    for column in [
        "revenue",
        "gross_margin",
        "ebit_margin",
        "sales_velocity",
        "revenue_change",
        "delta_revenue",
    ]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    summary = f"Scenario modeled on {len(df)} stores covering the last {days} day(s)."
    segments = _top_bottom_segments(df, "ebit_margin")
    plays = _scenario_plays(df, target_margin)
    plan = _thirty_day_plan(goal)

    return {
        "timeframe": f"last {days} days",
        "summary": summary,
        "segments": _sanitize_record(segments),
        "plays": _sanitize_record(plays),
        "plan": _sanitize_record(plan),
        "goal": goal,
        "target_margin": target_margin,
        "stores": _sanitize_record(df.to_dict(orient="records")),
    }
