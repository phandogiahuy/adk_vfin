from __future__ import annotations

import logging
import os
import random
from decimal import Decimal
from datetime import datetime
from pathlib import Path
from tempfile import SpooledTemporaryFile
from typing import Any, Dict, List, Literal, Optional

from dotenv import load_dotenv
from fastapi.encoders import jsonable_encoder
from google.adk import Agent
from google.adk.apps.app import App
from google.adk.tools import FunctionTool, transfer_to_agent

from my_agent.forecast_agent import forecast_next_days
from my_agent.nudge_agent import generate_nudges
from my_agent.scenario_ai import ai_scenario_advise
from my_agent import bq, insights
from my_agent.contracts import ContractIngestError, ingest_contract_document
from my_agent.ingest import ingest_dataset

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

logger = logging.getLogger("vfin.agent")

PROJECT_ID = os.getenv("PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT")
DATASET = os.getenv("BQ_DATASET") or os.getenv("DATASET", "vfin_sme")
MODEL_NAME = os.getenv("ADK_MODEL") or os.getenv("GEMINI_MODEL") or "gemini-2.5-pro"
AUDITOR_CODE = (
    os.getenv("ACCESS_AUDIT")
    or os.getenv("ACCESS_AUDITOR")
    or os.getenv("AUDITOR_ACCESS_CODE")
)
CEO_CODE = (
    os.getenv("ACCESS_CEO")
    or os.getenv("ACCESSCODE_CEO")
    or os.getenv("accesscode_ceo")
)


class LocalUploadFile:
    """Minimal UploadFile replacement so ingestion utilities work without FastAPI."""

    def __init__(self, buffer: SpooledTemporaryFile, filename: str, content_type: str):
        self.file = buffer
        self.filename = filename
        self.content_type = content_type

    async def close(self) -> None:
        self.file.close()


def _ensure_file(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"File '{path}' was not found.")
    if not path.is_file():
        raise ValueError(f"'{path}' is not a file.")
    return path


def _to_upload_file(path: Path, content_type: str) -> LocalUploadFile:
    buffer = SpooledTemporaryFile(max_size=1024 * 1024 * 50)
    with path.open("rb") as source:
        buffer.write(source.read())
    buffer.seek(0)
    return LocalUploadFile(buffer=buffer, filename=path.name, content_type=content_type)


async def upload_store_transactions(
    file_path: str,
    dataset_type: Literal["store_daily_transactions", "transactions"] = "store_daily_transactions",
) -> dict:
    """Loads daily store CSV data for employees into BigQuery."""

    path = _ensure_file(file_path)
    upload = _to_upload_file(path, "text/csv")
    try:
        result = await ingest_dataset(upload, dataset_type)
        return {
            "table": result.get("table"),
            "rows_ingested": result.get("rows"),
            "filename": result.get("filename"),
        }
    finally:
        await upload.close()


async def upload_audit_payload(
    file_path: str,
    dataset_type: Literal["audit_schedule", "compliance_incidents", "compliance_rules"],
) -> dict:
    """Loads auditor CSV packages into the appropriate compliance table."""

    path = _ensure_file(file_path)
    upload = _to_upload_file(path, "text/csv")
    try:
        result = await ingest_dataset(upload, dataset_type)
        return {
            "dataset_type": result.get("dataset_type"),
            "rows_ingested": result.get("rows"),
            "table": result.get("table"),
        }
    finally:
        await upload.close()


async def upload_contract_pdf(
    file_path: str, store_id: str, uploaded_by: Optional[str] = None
) -> dict:
    """Processes an employee PDF contract via Document AI and stores the insights."""

    path = _ensure_file(file_path)
    mime_type = "application/pdf"
    content = path.read_bytes()
    try:
        record = ingest_contract_document(
            content=content,
            mime_type=mime_type,
            store_id=store_id,
            filename=path.name,
            uploaded_by=uploaded_by or "employee",
        )
    except ContractIngestError as exc:
        raise ValueError(str(exc)) from exc
    return {
        "store_id": record.get("store_id"),
        "contract_summary": record.get("contract_summary"),
        "uploaded_at": record.get("uploaded_at"),
    }


def verify_access_code(role: Literal["auditor", "ceo"], provided_code: str) -> dict:
    """Validates access codes before exposing privileged auditor or CEO actions."""

    expected = AUDITOR_CODE if role == "auditor" else CEO_CODE
    ok = bool(expected and provided_code.strip() == expected.strip())
    return {
        "role": role,
        "authorized": ok,
        "message": "Access granted." if ok else "Access denied. Check the configured access code.",
    }


def _sanitize_value(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, dict):
        return {k: _sanitize_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_value(v) for v in value]
    return value


def _df_records(df: Any) -> List[Dict[str, Any]]:
    if df is None:
        return []
    if hasattr(df, "to_dict"):
        records = df.to_dict(orient="records")
    elif isinstance(df, list):
        records = df
    else:
        records = []
    return [_sanitize_value(record) for record in records]


def generate_executive_dashboard(days: int = 30) -> dict:
    """Builds a concise CEO dashboard using store transactions and alerts."""

    summary = bq.summary_by_region(days=days)
    stores = bq.store_financial_snapshot(days=days, limit=25)
    top = insights.top_stores(days=days, limit=3)
    bottom = insights.bottom_stores(days=days, limit=3)
    cash = insights.cash_health(days=days)
    alerts = insights.recent_alerts(limit=5)
    dashboard = {
        "timeframe_days": days,
        "regional_cash": _df_records(summary),
        "store_snapshot": _df_records(stores),
        "spotlight": {"top": top, "bottom": bottom},
        "cash_watch": cash,
        "latest_alerts": alerts,
    }
    return _sanitize_value(dashboard)


def _parse_date(date_str: str) -> datetime:
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    raise ValueError("Provide the date as YYYY-MM-DD.")


def _coerce_datetime(value: Any) -> Optional[datetime]:
    if value in (None, "", "null"):
        return None
    if isinstance(value, datetime):
        return value
    text = str(value)
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _coerce_whole_number(value: Optional[Any], field: str) -> Optional[int]:
    if value in (None, "", "null"):
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a number, not boolean.")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError(f"{field} must be a whole number.")
        return int(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        if "." in text:
            num = float(text)
            if not num.is_integer():
                raise ValueError
            return int(num)
        return int(text)
    except ValueError as exc:
        raise ValueError(f"{field} must be a whole number.") from exc


def _normalize_period_inputs(
    year: Optional[Any],
    month: Optional[Any],
    day: Optional[Any],
) -> tuple[Optional[int], Optional[int], Optional[int]]:
    return (
        _coerce_whole_number(year, "year"),
        _coerce_whole_number(month, "month"),
        _coerce_whole_number(day, "day"),
    )


def _normalize_limit(value: Optional[Any], default: int = 25) -> int:
    normalized = _coerce_whole_number(value, "limit")
    if normalized is None:
        return default
    return max(1, normalized)


def _validate_period_filters(
    year: Optional[int], month: Optional[int], day: Optional[int]
) -> None:
    if month is not None and year is None:
        raise ValueError("Provide a year before specifying a month.")
    if day is not None and (year is None or month is None):
        raise ValueError("Provide both year and month before specifying a day.")


def _describe_period(year: Optional[int], month: Optional[int], day: Optional[int]) -> str:
    if year is None:
        return "all available data"
    label = f"{year}"
    if month is not None:
        label += f"-{month:02d}"
        if day is not None:
            label += f"-{day:02d}"
    return label


def _filter_records_by_day(
    records: List[Dict[str, Any]],
    candidate_keys: List[str],
    year: Optional[int],
    month: Optional[int],
    day: Optional[int],
) -> List[Dict[str, Any]]:
    if day is None:
        return records
    filtered: List[Dict[str, Any]] = []
    for record in records:
        dt: Optional[datetime] = None
        for key in candidate_keys:
            value = record.get(key)
            if value is None:
                continue
            dt = _coerce_datetime(value)
            if dt:
                break
        if not dt:
            continue
        if year is not None and dt.year != year:
            continue
        if month is not None and dt.month != month:
            continue
        if dt.day != day:
            continue
        filtered.append(record)
    return filtered


def fetch_contract_compliance(
    year: Optional[float] = None,
    month: Optional[float] = None,
    day: Optional[float] = None,
    limit: Optional[int] = None,
) -> dict:
    """Retrieves contract_intelligence rows for the requested period."""

    normalized_year, normalized_month, normalized_day = _normalize_period_inputs(year, month, day)
    _validate_period_filters(normalized_year, normalized_month, normalized_day)
    normalized_limit = _normalize_limit(limit, 25)
    df = bq.contracts_intelligence(year=normalized_year, month=normalized_month, limit=normalized_limit)
    rows = _df_records(df)
    rows = _filter_records_by_day(rows, ["upload_date"], normalized_year, normalized_month, normalized_day)
    return {
        "filter": _describe_period(normalized_year, normalized_month, normalized_day),
        "records": rows,
    }


def _load_rules_snapshot(
    year: Optional[int], month: Optional[int], limit: int = 50
) -> List[Dict[str, Any]]:
    if not PROJECT_ID:
        return []
    try:
        from google.cloud import bigquery  # type: ignore
    except Exception as exc:  # pragma: no cover - informative logging
        logger.warning("BigQuery client unavailable: %s", exc)
        return []

    try:
        client = bigquery.Client(project=PROJECT_ID)
        filters: List[str] = []
        params: List[Any] = [bigquery.ScalarQueryParameter("limit", "INT64", limit)]
        if year is not None:
            filters.append("EXTRACT(YEAR FROM DATE(IFNULL(last_updated, CURRENT_DATE()))) = @year")
            params.append(bigquery.ScalarQueryParameter("year", "INT64", year))
        if month is not None:
            filters.append("EXTRACT(MONTH FROM DATE(IFNULL(last_updated, CURRENT_DATE()))) = @month")
            params.append(bigquery.ScalarQueryParameter("month", "INT64", month))
        where_clause = " AND ".join(filters) if filters else "TRUE"
        sql = f"""
            SELECT rule_id, category, title, severity, metric, threshold, responsible_role, remediation, last_updated
            FROM `{PROJECT_ID}.{DATASET}.compliance_rules`
            WHERE {where_clause}
            ORDER BY last_updated DESC
            LIMIT @limit
        """
        job_config = bigquery.QueryJobConfig(query_parameters=params)
        df = client.query(sql, job_config=job_config).result().to_dataframe()
        return df.to_dict(orient="records")
    except Exception as exc:  # pragma: no cover - informative logging
        logger.error("Failed to load compliance rules snapshot: %s", exc)
        return []


def fetch_operational_compliance(
    year: Optional[float] = None,
    month: Optional[float] = None,
    day: Optional[float] = None,
    limit: Optional[float] = None,
) -> dict:
    """Summarizes audits, incidents, scores, and rules filtered by the requested period."""

    normalized_year, normalized_month, normalized_day = _normalize_period_inputs(year, month, day)
    _validate_period_filters(normalized_year, normalized_month, normalized_day)
    normalized_limit = _normalize_limit(limit, 25)

    audits_df = bq.upcoming_audits(year=normalized_year, month=normalized_month, limit=normalized_limit)
    incidents_df = bq.compliance_incidents(
        open_only=False, year=normalized_year, month=normalized_month, limit=normalized_limit
    )
    scores_df = bq.compliance_scores(year=normalized_year, month=normalized_month, limit=normalized_limit)
    rules_rows = _load_rules_snapshot(normalized_year, normalized_month, limit=normalized_limit)

    result = {
        "filter": _describe_period(normalized_year, normalized_month, normalized_day),
        "audits": _df_records(audits_df),
        "incidents": _df_records(incidents_df),
        "scores": _df_records(scores_df),
        "rules": rules_rows,
    }
    return jsonable_encoder(result)


def get_today_nudges(days: int = 3) -> dict:
    """Returns the latest nudges generated by Vertex AI for CEO nudges flow."""

    return generate_nudges(days=days)


def run_scenario_simulation(
    days: int = 30, target_margin: Optional[float] = None, goal: Optional[str] = None
) -> dict:
    """Calls the scenario advisor so CEOs can simulate next steps."""

    result = ai_scenario_advise(days=days, target_margin=target_margin, goal=goal)
    return _sanitize_value(result)


def run_predictive_forecast(days: int = 7) -> dict:
    """Runs the predictive BigQuery ML + macro adjusted forecast for CEOs."""

    if days < 1 or days > 30:
        raise ValueError("Days must be between 1 and 30.")

    return forecast_next_days(days=days)


def fun_signoff(role: Literal["employee", "auditor", "ceo"] = "employee") -> dict:
    """Returns a random playful farewell that matches the persona requested by the flow."""

    messages = {
        "employee": [
            "Data is safely uploaded—grab a coffee and conquer the next shift!",
            "Books are tidy. Hope the rest of your shift runs smoothly!",
            "Mission accomplished. Time to tally inventory like a pro.",
        ],
        "auditor": [
            "Checklists closed, compliance smiling—stretch before the next review!",
            "Risk radar is calm. Treat yourself to an audit-friendly tea!",
            "Standards secured. Keep that laser focus for the next inspection.",
        ],
        "ceo": [
            "Dashboard delivered. Wishing you bold, brilliant decisions!",
            "Alerts addressed—now it's time to launch the next winning play.",
            "All signals summarized. Don't forget to breathe before the board call!",
        ],
    }
    choice = random.choice(messages.get(role, messages["employee"]))
    return {"role": role, "message": choice}


CONTRACT_REVIEW_FORMAT = """
You are a senior retail executive and contract-risk specialist.
Generate a Contract Review Report in the exact structure below.
Do NOT use tables. Use clear bullet points and short paragraphs.
The report period will be: [daily / monthly / yearly — insert here].

Required Structure:

1. Summary for Leadership
Give a brief overview of the contract status for the selected period: main risks, key financial impact, and the top 2–3 decisions leadership must consider.

2. Store-by-Store Assessment
For each store, provide:

A short health rating (e.g., “High risk”, “Moderate risk”, “Low risk”)

One or two reasons why (contract terms, pricing, renewal, penalties, compliance gaps)

3. Major Red Flags
List the most critical clauses or operational risks that require immediate attention during the selected period. Keep each point short and decisive.

4. Financial & Operational Impact
Summarize potential consequences if issues persist: cost increases, margin pressure, supply risk, termination penalties, or service downtime.

5. Actions Required
List what needs to be fixed, who should own it (Legal / Procurement / Operations), and the expected timeframe.

6. Direction for Next Period
State what leadership should monitor in the next day/month/year: upcoming renewals, renegotiation opportunities, or systemic contract issues.

Additional Rules:

Keep the entire report concise and actionable.

No tables. No markdown tables. No code blocks.

Tone must be C-suite level: direct, strategic, decision-focused.

Reflect only the period provided (daily / monthly / yearly).
""".strip()


FORECAST_REPORT_FORMAT = """
You are a senior retail executive and forecasting specialist.
Create a Sales Forecast Report for all stores.
The forecast period is: [insert number of days].
Do NOT use tables. Use short sections and bullet points only.
Keep the tone concise and C-suite level.

Required Structure:

1. Executive Summary
Provide a brief high-level view of the forecast for the selected period: expected growth or decline, standout stores, and main drivers behind the trend.

2. Forecast by Store
For each store, provide:

Total forecasted sales for the chosen period

Expected daily trend (e.g., upward, stable, downward)

One short reason explaining the trend
Format example:
• Store HCM-01 — expected to increase slightly due to weekend traffic boost.
• Store HCM-02 — projected to decline; reduced footfall after promotion cycle.

3. Key Drivers
List the main factors influencing the forecast (seasonality, promotions, weather, inventory, market events, competitor movements).

4. Risks & Opportunities
Bullet-point potential risks (stockouts, staffing gaps, local events) and opportunities (high-margin products, upcoming campaigns).

5. Action Recommendations
Provide short, actionable steps for operations, inventory planning, or marketing to improve the forecast outcome.

6. Next-Period Focus
State what leadership should monitor in the next forecasting cycle and any critical changes to track.

Additional Notes:

Keep the entire report compact and business-focused.

Do NOT use tables, code blocks, or markdown grids.

Maintain clarity and decision orientation for leadership.
""".strip()


CEO_NUDGE_REPORT_FORMAT = """
You are a strategy advisor for the CEO.
Generate a short, action-driven “CEO Nudge Report” using the exact structure below.
Do NOT use tables. Use sharp, decisive bullet points.
This report must highlight immediate actions the CEO should take based on the latest business context provided.

Required Structure:

1. Immediate Priority Signals
List the 3–5 most urgent issues or opportunities that require CEO-level attention right now. Keep each point under two lines and focused on impact.

2. Critical Nudges for Action
Provide direct, non-negotiable nudges such as:
• “Decide whether to…”
• “Approve…”
• “Intervene in…”
• “Trigger negotiation on…”
Write each as a clear action the CEO must take immediately.

3. Fast Wins (High Impact, Low Effort)
Identify a few quick moves the CEO can execute within days to create momentum, reduce risk, or unlock growth.

4. Strategic Watchpoints
Highlight what the CEO should monitor closely in the next week/month: market signals, operational risks, financial shifts, or leadership alignment issues.

5. Consequence of Inaction
State in 2–3 bullets what will happen if the CEO does not act on these nudges.

Rules:

No tables.

No long paragraphs.

Tone must be urgent, decisive, and C-suite level.

Focus only on actions from the CEO, not from middle management.
""".strip()


DASHBOARD_REPORT_FORMAT = """
You are an executive intelligence assistant preparing a Morning Briefing for the CEO.
Generate a concise but high-impact daily report using the structure below.
Do NOT use tables. Use short sections and sharp bullet points only.
Tone must be strategic, urgent, and decision-oriented.

Required Structure:

1. Overnight Highlights
Summarize the most important developments since the previous day: market signals, sales movement, operational incidents, or urgent messages from key partners.

2. Business Performance Snapshot
Provide quick, narrative-only insights on sales, traffic, margin pressure, inventory status, or any early-warning trends. Keep it short, actionable, and avoid numbers unless meaningful.

3. Top 3 Risks to Watch Today
Bullet-point the key operational or financial risks that may impact the day. Keep each point under two lines.

4. Opportunities & Levers for Today
Highlight short-term opportunities the CEO can activate or approve today (pricing moves, partnership calls, marketing triggers, inventory unlocking, etc.).

5. People & Organization Signals
Mention any leadership issues, team capacity concerns, escalations, or cultural signals requiring CEO awareness.

6. Nudges for CEO Action
List the decisive actions or approvals the CEO should consider taking within the day. Each point must be phrased as a direct nudge (e.g., “Approve…”, “Call…”, “Greenlight…”).

7. Forward View (Next 48–72 Hours)
Provide a compact lookahead of what’s coming: deadlines, major meetings, potential market shifts, or any expected operational hotspots.

Rules:

No tables.

No long paragraphs.

Must be readable in under 60 seconds.

Must guide CEO action, not just provide information.
""".strip()


OPERATIONAL_COMPLIANCE_FORMAT = """
You are an operations and compliance auditor.
Generate an Operational Compliance Review for the period: [daily / monthly / yearly].
Do NOT use tables. Use short sections and bullet points.
Tone must be concise, factual, and C-suite appropriate.

Required Structure:

1. Compliance Overview for the Selected Period
Summarize overall adherence to operational standards, SOPs, safety requirements, and regulatory obligations. Highlight whether compliance is improving, stable, or declining.

2. Key Strengths Observed
List the areas that show solid compliance performance (process discipline, documentation quality, audit readiness, safety practices, staff training, etc.).

3. Major Non-Compliance Findings
Provide the most critical issues found during this period.
Keep each point short and specific (e.g., procedural gaps, missing logs, deviations, safety lapses, hygiene issues, incomplete records).

4. Root Causes (High-Level)
Summarize the likely underlying causes (training gaps, staffing constraints, system issues, unclear SOPs, equipment constraints, or leadership oversight gaps).

5. Operational Impact
Explain briefly how these compliance issues could affect operations: risks to quality, safety, productivity, customer experience, or regulatory exposure.

6. Required Corrective Actions
List the actions that must be taken now. Indicate responsible functions (Operations / QA / HR / Compliance / Maintenance) and short expected timelines.
Do not use table format — each action as a bullet.

7. Follow-Up & Monitoring Priorities
Highlight what leadership must monitor next (e.g., specific SOPs, high-risk teams, repeated deviations, audit checkpoints, equipment reliability).

8. Forecast & Risk Outlook
Provide a short prediction of compliance trends for the next cycle (day/month/year): improvement likelihood, potential hotspots, and any early warnings.

Rules:

No tables or grids.

Keep the report short, structured, and decision-oriented.

Remain neutral, analytical, and operations-focused.
""".strip()


SCENARIO_SIMULATION_FORMAT = """
You are a senior strategy advisor for the CEO.
Generate a scenario simulation based on the following target adjustment: “[insert goal, e.g., increase EBIT by 15%].”
Do NOT use tables. Use short sections and bullet points.
Keep the tone strategic, realistic, and CEO-ready.

Required Structure:

1. Scenario Summary
Provide a concise overview of what achieving this target would mean for the business.

2. Best Case – Base Case – Worst Case
For each case, provide:

Expected outcome

Key assumptions

Main risks or constraints
Use bullets only, no tables. Keep it sharp.

3. Required Levers to Reach the Target
List the strategic levers needed (pricing, cost structure, productivity, store performance, supply chain, product mix, opex management).

4. Trade-Offs & Consequences
Explain what the company would need to sacrifice or risk (margin compression, operational strain, competitive impact, talent pressure, customer experience effects).

5. Risks Under This Scenario
Highlight the major operational, financial, and organizational risks that emerge if the CEO pursues this target.

6. Alternative Paths to Achieve the Same Target
Provide 2–3 concise strategic routes (e.g., revenue-driven, cost-driven, structural-efficiency-driven), each with its own pros and cons.

7. CEO Decision Points
List the immediate decisions the CEO must make to activate the selected scenario.

8. Forward Outlook
Give a short prediction of how this choice will influence the next quarter or year.

Rules:

No tables.

No long paragraphs.

Keep everything high-impact, condensed, and actionable.
""".strip()

FINANCIAL_COMPANION_INSTRUCTION = """
You are a Financial Companion Agent for the CEO.
You act as both a professional financial advisor and a strategic thought partner.
Communicate clearly, confidently, and with C-suite-level insight, while keeping a supportive, collaborative tone.
The CEO may ask anything related to finance, strategy, forecasts, investments, operational impact, valuation, risk management, macroeconomics, or industry trends.
Use internal/company data when available. If the information is not in the database, acknowledge it and provide the most reliable, up-to-date external sources.
Always explain your reasoning, outline risks, and provide actionable recommendations.
Keep answers concise but insightful, focused on decision-making and real impact.
Your goal: help the CEO think better, see blind spots, and make smarter financial decisions.
""".strip()

ROOT_INSTRUCTION = """
You are the coordinator for the vFin SME ADK system. Your responsibilities:
1. Greet the user in English and explain that you can route them to one of three roles: Employee, Auditor, CEO.
2. Ask which role they want. If unsure, remind them what each role can do.
3. When they confirm, use transfer_to_agent to move them to the matching specialist agent:
   - employee_ops handles store-data uploads and contract ingestion.
   - auditor_ops covers audit schedule/incidents/rules with access-code gating.
   - ceo_ops delivers dashboards, compliance insights, nudges, simulations, and forecasts.
4. If they come back after finishing, help them jump to another role or summarize progress.
5. If they stop, offer a friendly farewell and close the conversation.
""".strip()


EMPLOYEE_INSTRUCTION = """
You serve employees running physical stores:
- Confirm which store/team they represent and whether they want to upload daily store data (CSV) or upload a contract (PDF).
- CSV uploads: remind them the file must contain store_id, store_name, region, province, date, revenue, cogs, and the usual cost/cash columns. Ask for a file path and call upload_store_transactions.
- Contract uploads: ask which store the contract belongs to, request a PDF path, then call upload_contract_pdf so Document AI processes it into contract_intelligence.
- After every action, echo which table/rows were updated and ask if they want the other action. If not, call fun_signoff(role="employee") for a cheerful goodbye.
""".strip()


AUDITOR_INSTRUCTION = """
You partner with auditors on compliance tasks:
- Always start by asking for the access code and call verify_access_code(role="auditor"). If it fails, politely ask again.
- Once verified, list the options as bullets: Upload audit schedule, Upload compliance incident, Upload compliance rules. Mention which BigQuery table each option feeds.
- When the auditor picks one, request the CSV path and call upload_audit_payload with the right dataset_type. Report the resulting row count/table.
- After each upload, ask whether they want to handle another option or finish. If they finish, trigger fun_signoff(role="auditor") with an upbeat tone.
""".strip()


CEO_INSTRUCTION = f"""
You are the CEO's digital chief of staff:
- Ask for the CEO access code via verify_access_code(role="ceo"). Stay polite until it's correct.
- Once validated, immediately call generate_executive_dashboard (30-day default) and rewrite the response using this Morning Briefing template (no tables):
{DASHBOARD_REPORT_FORMAT}
- Then ask what the CEO wants next and list the options: view contract compliance, view operational compliance, review nudges, simulate scenarios, or run predictive models. Each response must follow its specific format:
  * Contract compliance: confirm whether they want all data or filters for year/month/day (year required before month, month before day). Call fetch_contract_compliance(year=?, month=?, day=?). Rewrite the output exactly using this format:
{CONTRACT_REVIEW_FORMAT}
  * Operational compliance: mirror the same filter flow and call fetch_operational_compliance(year=?, month=?, day=?). Present the findings using this format:
{OPERATIONAL_COMPLIANCE_FORMAT}
  * Nudges: call get_today_nudges and convert the payload into this CEO Nudge Report format:
{CEO_NUDGE_REPORT_FORMAT}
  * Scenario simulation: gather the target goal, call run_scenario_simulation, and rewrite the answer using this scenario format:
{SCENARIO_SIMULATION_FORMAT}
  * Predictive model: ask for a 1-30 day horizon, call run_predictive_forecast, and present the results via this forecast format:
{FORECAST_REPORT_FORMAT}
- If the CEO asks broader financial/strategic questions outside the flows above, call transfer_to_agent("ceo_companion") to route them to the Financial Companion agent.
- After each response, offer other choices. If they are done, call fun_signoff(role="ceo") with a polished, upbeat farewell.
""".strip()


employee_agent = Agent(
    name="employee_ops",
    model=MODEL_NAME,
    instruction=EMPLOYEE_INSTRUCTION,
    description="Helps store employees upload daily data and contracts into BigQuery.",
    tools=[
        FunctionTool(upload_store_transactions),
        FunctionTool(upload_contract_pdf),
        FunctionTool(fun_signoff),
    ],
)


auditor_agent = Agent(
    name="auditor_ops",
    model=MODEL_NAME,
    instruction=AUDITOR_INSTRUCTION,
    description="Supports auditors with schedules, incidents, and rules management.",
    tools=[
        FunctionTool(verify_access_code),
        FunctionTool(upload_audit_payload),
        FunctionTool(fun_signoff),
    ],
)

ceo_companion_agent = Agent(
    name="ceo_companion",
    model=MODEL_NAME,
    instruction=FINANCIAL_COMPANION_INSTRUCTION,
    description="Handles open-ended CEO financial and strategic questions.",
    tools=[
        FunctionTool(verify_access_code),
        FunctionTool(generate_executive_dashboard),
        FunctionTool(fetch_contract_compliance),
        FunctionTool(fetch_operational_compliance),
        FunctionTool(get_today_nudges),
        FunctionTool(run_scenario_simulation),
        FunctionTool(run_predictive_forecast),
        FunctionTool(fun_signoff),
    ],
)

ceo_agent = Agent(
    name="ceo_ops",
    model=MODEL_NAME,
    instruction=CEO_INSTRUCTION,
    description="Executive assistant delivering dashboards, compliance views, nudges, simulations, and forecasts.",
    tools=[
        transfer_to_agent,
        FunctionTool(verify_access_code),
        FunctionTool(generate_executive_dashboard),
        FunctionTool(fetch_contract_compliance),
        FunctionTool(fetch_operational_compliance),
        FunctionTool(get_today_nudges),
        FunctionTool(run_scenario_simulation),
        FunctionTool(run_predictive_forecast),
        FunctionTool(fun_signoff),
    ],
)


root_agent = Agent(
    name="orchestrator",
    model=MODEL_NAME,
    instruction=ROOT_INSTRUCTION,
    description="Greets users and routes them to the best-fit specialist agent.",
    tools=[transfer_to_agent],
    sub_agents=[employee_agent, auditor_agent, ceo_agent, ceo_companion_agent],
)
