from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List

from io import BytesIO

from google.api_core.client_options import ClientOptions
from google.cloud import documentai_v1 as documentai

from . import bq

PROJECT_ID = os.getenv("DOCUMENTAI_PROJECT")
PROCESSOR_LOCATION = os.getenv("CONTRACT_PROCESSOR_LOCATION", "us")
PROCESSOR_ID = os.getenv("CONTRACT_PROCESSOR_ID")
CONTRACT_TABLE = os.getenv("CONTRACT_TABLE", "contract_intelligence")


class ContractIngestError(RuntimeError):
    """Raised when a contract cannot be processed."""


def _docai_client():
    if not PROJECT_ID or not PROCESSOR_ID:
        raise ContractIngestError("Document AI processor is not configured.")
    try:
        api_endpoint = f"{PROCESSOR_LOCATION}-documentai.googleapis.com"
        client_options = ClientOptions(api_endpoint=api_endpoint)
        return documentai.DocumentProcessorServiceClient(client_options=client_options)
    except Exception as exc:
        raise ContractIngestError(f"Unable to initialize Document AI client: {exc}") from exc


def _safe_float(value: str | None) -> float | None:
    if not value:
        return None
    try:
        cleaned = value.replace(",", "").replace("$", "")
        return float(cleaned)
    except Exception:
        return None


def _safe_date(value: str | None) -> str | None:
    if not value:
        return None
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(value, fmt).date().isoformat()
        except Exception:
            continue
    return value


def _extract_contract(content: bytes, mime_type: str) -> Dict[str, Any]:
    try:
        client = _docai_client()
        processor_name = client.processor_path(PROJECT_ID, PROCESSOR_LOCATION, PROCESSOR_ID)
        raw_document = documentai.RawDocument(content=content, mime_type=mime_type)
        request = documentai.ProcessRequest(name=processor_name, raw_document=raw_document)
        result = client.process_document(request=request)
        document = result.document
    except Exception as exc:
        raise ContractIngestError(f"Document AI processing failed: {exc}") from exc
    record: Dict[str, Any] = {
        "contract_id": None,
        "store_id": None,
        "contract_summary": document.text,
    }

    def add_flag(flag: str) -> None:
        flags: List[str] = record.setdefault("risk_flags", [])
        if flag not in flags:
            flags.append(flag)

    for entity in document.entities:
        text = (entity.mention_text or "").strip()
        if not text:
            continue
        etype = (entity.type_ or "").lower()
        if etype in {"contract_number", "contract_id", "document_id"}:
            record["contract_id"] = text
        elif etype in {"party", "counterparty", "vendor", "supplier"}:
            record["counterparty"] = text
        elif etype in {"store_id", "store"}:
            record["store_id"] = text
        elif etype in {"total_amount", "contract_value", "amount"} or "value" in etype:
            value = _safe_float(text)
            if value is not None:
                record["contract_value"] = value
        elif etype in {"currency"}:
            record["contract_currency"] = text
        elif etype in {"effective_date", "start_date"} or "commence" in etype:
            record["start_date"] = _safe_date(text)
        elif etype in {"end_date", "expiry_date", "expiration_date"}:
            record["end_date"] = _safe_date(text)
        elif "payment" in etype and "term" in etype:
            record["payment_terms"] = text
        elif "termination" in etype:
            record["termination_clause"] = text
            if "auto" in text.lower():
                add_flag("Automatic termination clause detected.")
        elif "penalty" in etype or "liability" in etype:
            record["penalty_clause"] = text
        elif "renewal" in etype:
            add_flag("Auto-renewal clause identified.")

        if entity.confidence and entity.confidence < 0.6:
            add_flag(f"Low confidence on {etype}")

    if not record["contract_id"]:
        record["contract_id"] = str(uuid.uuid4())

    return record


def _normalize_document(content: bytes, mime_type: str) -> tuple[bytes, str]:
    """Convert unsupported mime types (e.g., PNG/JPG) into PDF bytes."""

    normalized_mime = (mime_type or "").lower()
    if normalized_mime in {"application/pdf", "application/octet-stream"}:
        return content, "application/pdf"

    if normalized_mime in {"image/png", "image/jpeg", "image/jpg"}:
        try:
            from PIL import Image  # type: ignore
        except ImportError as exc:
            raise ContractIngestError("Pillow is required to process image contracts.") from exc

        with BytesIO(content) as buf:
            try:
                image = Image.open(buf)
            except Exception as exc:
                raise ContractIngestError("Unable to open image for conversion.") from exc
            rgb = image.convert("RGB")
            output = BytesIO()
            rgb.save(output, format="PDF")
            return output.getvalue(), "application/pdf"

    raise ContractIngestError(f"Unsupported contract mime type: {mime_type or 'unknown'}")


def ingest_contract_document(
    content: bytes,
    mime_type: str,
    store_id: str | None,
    filename: str | None,
    uploaded_by: str | None = None,
) -> Dict[str, Any]:
    if not content:
        raise ContractIngestError("Empty file uploaded.")
    normalized_content, normalized_mime = _normalize_document(content, mime_type)
    record = _extract_contract(normalized_content, normalized_mime)
    record["store_id"] = store_id or record.get("store_id") or "UNKNOWN"
    record["filename"] = filename
    record["uploaded_by"] = uploaded_by or "employee"
    record["uploaded_at"] = datetime.now(timezone.utc).isoformat()
    bq.insert_contract_record(record, table=CONTRACT_TABLE)
    return record
