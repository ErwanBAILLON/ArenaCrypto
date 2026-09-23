"""Universe of tradable symbols and fee assumptions, loaded from config/universe.yaml."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import yaml

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "config" / "universe.yaml"


@dataclass(frozen=True)
class Fees:
    perp_taker: float
    slippage: float
    spot_taker: float


@dataclass(frozen=True)
class Membership:
    """Point-in-time universe rules. When enabled, ``Universe.symbols`` is only a superset."""

    enabled: bool = False
    top_n: int = 50
    min_adv_usd: float = 5_000_000.0
    window_days: int = 30


@dataclass(frozen=True)
class Impact:
    """Per-symbol execution cost. Disabled means the flat slippage, byte-identical to before."""

    enabled: bool = False
    half_spread: float = 0.0002
    k: float = 1.0
    capacity_nav: float = 1_000_000.0


@dataclass(frozen=True)
class Universe:
    symbols: list[str]
    binance_suffix: str
    fees: Fees
    nav0: float
    history_start: datetime
    name: str = "crypto"
    exchange: str = "binance"  # data source: "binance" (1h perps) or "yahoo" (1d classic markets)
    bar: str = "1h"  # "1h" or "1d": the decision bar of this universe
    membership: Membership = field(default_factory=Membership)
    impact: Impact = field(default_factory=Impact)

    @property
    def reference(self) -> str:
        """Symbol used for regime labelling and warm-up: BTC when present, else the first symbol."""
        return "BTC" if "BTC" in self.symbols else self.symbols[0]

    @property
    def bar_hours(self) -> int:
        return {"1h": 1, "1d": 24}[self.bar]

    @property
    def bars_per_day(self) -> int:
        return 24 // self.bar_hours

    def binance_symbol(self, symbol: str) -> str:
        return f"{symbol}{self.binance_suffix}"

    def impact_model(self):
        """The configured ``ImpactModel``, or None when this arena uses flat costs."""
        from arena.core.costs import ImpactModel

        if not self.impact.enabled:
            return None
        return ImpactModel(half_spread=self.impact.half_spread, k=self.impact.k, capacity_nav=self.impact.capacity_nav)

    def membership_rule(self):
        from arena.core.membership import MembershipRule

        return MembershipRule(
            top_n=self.membership.top_n,
            min_adv_usd=self.membership.min_adv_usd,
            window_days=self.membership.window_days,
        )

    def hyperliquid_coin(self, symbol: str) -> str:
        return symbol


def load_universe(path: str | Path | None = None) -> Universe:
    p = Path(path) if path else DEFAULT_PATH
    raw = yaml.safe_load(p.read_text())
    fees = raw.get("fees", {})
    start = raw.get("history_start", "2024-01-01T00:00:00Z")
    ts = datetime.fromisoformat(str(start).replace("Z", "+00:00"))
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
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
        name=str(raw.get("name", "crypto")),
        exchange=str(raw.get("exchange", "binance")),
        bar=str(raw.get("bar", "1h")),
        membership=Membership(**{k: v for k, v in (raw.get("membership") or {}).items()}),
        impact=Impact(**{k: v for k, v in (raw.get("impact") or {}).items()}),
    )
