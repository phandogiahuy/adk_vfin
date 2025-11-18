from __future__ import annotations

import datetime as dt
import os
import time
from typing import Dict, List, Optional

import requests

WORLD_BANK_BASE = "https://api.worldbank.org/v2/country/VNM/indicator/{code}"
INDICATORS = {
    "gdp_growth": {
        "code": "NY.GDP.MKTP.KD.ZG",
        "label": "GDP growth (annual %)",
        "baseline": 6.0,
        "weight": 0.6,
    },
    "inflation_rate": {
        "code": "FP.CPI.TOTL.ZG",
        "label": "Inflation rate (consumer prices, %)",
        "baseline": 3.0,
        "weight": -0.4,
    },
    "cpi_index": {
        "code": "FP.CPI.TOTL",
        "label": "Consumer price index (2010=100)",
        "baseline": 110.0,
        "weight": -0.2,
    },
}

_CACHE_TTL_SECONDS = int(os.getenv("MACRO_CACHE_TTL", "43200"))  # 12h
_CACHE: Dict[str, tuple[float, Dict[str, Dict[str, Optional[float]]]]] = {}


def _fetch_indicator(code: str, start_year: int, end_year: int) -> List[dict]:
    params = {
        "format": "json",
        "per_page": 200,
        "date": f"{start_year}:{end_year}",
    }
    url = WORLD_BANK_BASE.format(code=code)
    resp = requests.get(url, params=params, timeout=90)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, list) or len(data) < 2:
        return []
    return data[1] or []


def _latest_value(series: List[dict]) -> tuple[Optional[float], Optional[str]]:
    for entry in series:
        value = entry.get("value")
        if value is not None:
            return float(value), entry.get("date")
    return None, None


def _load_macro_snapshot() -> Dict[str, Dict[str, Optional[float]]]:
    end_year = dt.datetime.timezone.utc.now().year
    start_year = end_year - 5
    snapshot: Dict[str, Dict[str, Optional[float]]] = {}
    for key, meta in INDICATORS.items():
        series = _fetch_indicator(meta["code"], start_year, end_year)
        value, period = _latest_value(series)
        snapshot[key] = {
            "label": meta["label"],
            "value": value,
            "period": period,
            "baseline": meta["baseline"],
            "weight": meta["weight"],
        }
    return snapshot


def get_macro_snapshot() -> Dict[str, Dict[str, Optional[float]]]:
    now = time.time()
    expires, cached = _CACHE.get("snapshot", (0.0, None))
    if cached is None or now > expires:
        cached = _load_macro_snapshot()
        _CACHE["snapshot"] = (now + _CACHE_TTL_SECONDS, cached)
    return cached


def compute_macro_factor(snapshot: Dict[str, Dict[str, Optional[float]]]) -> tuple[float, Dict[str, float]]:
    contributions: Dict[str, float] = {}
    total_adjustment = 0.0
    for key, meta in INDICATORS.items():
        data = snapshot.get(key, {})
        value = data.get("value")
        baseline = data.get("baseline")
        weight = data.get("weight", 0.0)
        if value is None or baseline in (None, 0):
            continue
        # Normalized deviation (percentage for growth rates, relative for index)
        if key == "cpi_index":
            deviation = (value - baseline) / baseline
        else:
            deviation = (value - baseline) / 100.0
        contribution = deviation * weight
        contributions[key] = contribution
        total_adjustment += contribution
    macro_factor = max(0.85, min(1.15, 1.0 + total_adjustment))
    return macro_factor, contributions
