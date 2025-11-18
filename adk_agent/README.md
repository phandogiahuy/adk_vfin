# vFin SME ADK Agent

Agent bundle located in `my_agent/` built with Google ADK. It contains the orchestrator, employee/auditor/CEO flows, and a financial companion agent, plus supporting tools for BigQuery ingest, Document AI contract processing, nudges, forecasts, and scenario simulation.

---

## 1. Prerequisites

- Python 3.11+ and `pip`.
- Google Cloud SDK (`gcloud`) and Docker (required for Cloud Run deployments).
- Enabled GCP APIs: Cloud Run, Cloud Build, Artifact Registry, Vertex AI, BigQuery, Document AI.
- Service account permissions: BigQuery Admin, Vertex AI User, Cloud Run Admin, Artifact Registry Writer, Cloud Build Service Account, Document AI (processor) access.

---

## 2. Local Environment Setup

```powershell
git clone <repo-url>
cd adk_agent
python -m venv .venv
.\.venv\Scripts\activate    # PowerShell (use `source .venv/bin/activate` on macOS/Linux)
pip install -r requirements.txt
```


---

## 3. Project Highlights

- `my_agent/agent.py`: root orchestrator plus employee, auditor, CEO, and financial companion agents. Contains tool wiring, prompts, and instructions.
- `my_agent/services/*.py`: BigQuery utilities, insights, Document AI ingestion, macro calculations, etc.
- `my_agent/ingest.py`: accepts CSV/XLSX for transactions, audit schedules, compliance incidents/rules. `dataset_type` supports `"transactions"` and `"store_daily_transactions"` alias.
- `my_agent/nudge_agent.py`: deterministic nudges for the CEO flow (no LLM prompt needed).
- `my_agent/scenario_ai.py`: scenario simulation built directly from store metrics (no LLM prompt). 
- `my_agent/forecast_agent.py`: BigQuery ML forecast with World Bank macro adjustment.
- `my_agent/macro.py`: fetches macro indicators with 90s timeout and caching.

---

## 4. Running Locally

1. Activate venv + env vars.
2. Launch ADK web UI:
   ```powershell
   adk web my_agent
   ```
   Access via `http://localhost:8080`.
3. CLI mode:
   ```powershell
   adk run my_agent --host=0.0.0.0 --port=8080
   ```
4. Upload flows: when prompted for files, always provide a real path (e.g., `C:\Users\Admin\Documents\transactions.xlsx`). CSV/XLSX are auto-detected.

---

## 5. Testing Operational Compliance Fix

To verify `fetch_operational_compliance` works (no Pydantic float serialization issues):
```powershell
python - <<'PY'
from my_agent import agent
print(agent.fetch_operational_compliance(year="2025"))
PY
```
Returns a JSON-friendly dict sanitized via `jsonable_encoder`.

---

## 6. Deploying to Cloud Run

### Manual Docker Path
1. Ensure `gcloud` and `docker` run from PowerShell (`gcloud --version`, `docker --version`).
2. Build/push image:
   ```powershell
   gcloud config set project $Env:GOOGLE_CLOUD_PROJECT
   gcloud builds submit --tag us-central1-docker.pkg.dev/$Env:GOOGLE_CLOUD_PROJECT/adk-repo/my-agent:v1 .
   ```
3. Deploy:
   ```powershell
   gcloud run deploy my-agent-service `
     --image us-central1-docker.pkg.dev/$Env:GOOGLE_CLOUD_PROJECT/adk-repo/my-agent:v1 `
     --region $Env:GOOGLE_CLOUD_LOCATION `
     --allow-unauthenticated
   ```

### ADK CLI Shortcut (automatic packaging)
Ensure `gcloud` is in PATH (Windows fix uses `gcloud.cmd`). Then:
```powershell
adk deploy cloud_run --project=$Env:GOOGLE_CLOUD_PROJECT --region=$Env:GOOGLE_CLOUD_LOCATION my_agent
```
The CLI generates a temp Dockerfile, runs Cloud Build + Cloud Run deploy. If you need the UI deployed, add `--with_ui`. Example:
```powershell
adk deploy cloud_run --project=$Env:GOOGLE_CLOUD_PROJECT --region=$Env:GOOGLE_CLOUD_LOCATION --service_name=$Env:SERVICE_NAME --app_name=$Env:APP_NAME --with_ui my_agent
```

If you hit `[WinError 2]` ensure the updated CLI file (`cli_deploy.py`) is in place or update ADK to a version with the Windows fix.

---

## 7. Financial Companion Agent

- Added as a sub-agent of the CEO flow (`ceo_companion`). Handles open-ended financial/strategic questions using the defined prompt. The CEO instruction now routes off-script queries via `transfer_to_agent("ceo_companion")`.

---

## 8. Troubleshooting

- **Upload issues**: always provide absolute file paths. The agent can’t ingest raw text.
- **Operational compliance**: serialization errors fixed by `jsonable_encoder` in `fetch_operational_compliance`.
- **Document AI**: requires `CONTRACT_PROCESSOR_ID`, `CONTRACT_PROCESSOR_LOCATION`, and `DOCUMENTAI_PROJECT` env vars.
- **BigQuery**: ensure dataset/table exist and service account has write access.
- **gcloud errors on Windows**: use PowerShell backticks (`) instead of `\`, and ensure `gcloud.cmd` is on PATH.

---

With the steps above, anyone can configure the environment, run the agent locally, and deploy to Cloud Run. Update this README as flows or deployment steps evolve.
