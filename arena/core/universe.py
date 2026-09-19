"""Universe of tradable symbols and fee assumptions, loaded from config/universe.yaml."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import yaml

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "config" / "universe.yaml"


@dataclass(frozen=True)
class Fees:
    perp_taker: float
    slippage: float
    spot_taker: float


@dataclass(frozen=True)
class Universe:
    symbols: list[str]
    binance_suffix: str
    fees: Fees
    nav0: float
    history_start: datetime

    def binance_symbol(self, symbol: str) -> str:
        return f"{symbol}{self.binance_suffix}"

    def hyperliquid_coin(self, symbol: str) -> str:
        return symbol


def load_universe(path: str | Path | None = None) -> Universe:
    p = Path(path) if path else DEFAULT_PATH
    raw = yaml.safe_load(p.read_text())
    fees = raw.get("fees", {})
    start = raw.get("history_start", "2024-01-01T00:00:00Z")
    ts = datetime.fromisoformat(str(start).replace("Z", "+00:00"))
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return Universe(
        symbols=list(raw["symbols"]),
        binance_suffix=str(raw.get("binance_suffix", "USDT")),
        fees=Fees(
            perp_taker=float(fees.get("perp_taker", 0.0005)),
            slippage=float(fees.get("slippage", 0.0002)),
            spot_taker=float(fees.get("spot_taker", 0.0010)),
        ),
        nav0=float(raw.get("nav0", 10_000.0)),
        history_start=ts,
    )
