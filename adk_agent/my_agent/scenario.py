from __future__ import annotations

from pydantic import BaseModel
from typing import Dict


class ScenarioInput(BaseModel):
    revenue: float
    cogs: float
    opex: float
    rev_pct: float = 0
    cogs_pct: float = 0
    opex_pct: float = 0


def run_scenario(data: ScenarioInput) -> Dict[str, float]:
    rev_new = data.revenue * (1 + data.rev_pct / 100.0)
    cogs_new = data.cogs * (1 + data.cogs_pct / 100.0)
    opex_new = data.opex * (1 + data.opex_pct / 100.0)

    gross_profit = rev_new - cogs_new
    gross_margin = 0.0 if rev_new == 0 else gross_profit / rev_new
    ebit = gross_profit - opex_new
    cm = max(gross_margin, 0.0001)
    bep_days = (opex_new / (cm * (rev_new if rev_new > 0 else 1))) if rev_new > 0 else 365

    return {
        "revenue_new": rev_new,
        "cogs_new": cogs_new,
        "expenses_new": opex_new,
        "gross_profit": gross_profit,
        "gross_margin": gross_margin,
        "ebit": ebit,
        "bep_days": min(bep_days, 365.0),
    }

