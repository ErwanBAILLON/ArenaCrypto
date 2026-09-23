import time

import pandas as pd
import pytest

from arena.book.book import FeeModel
from arena.core.types import Target
from arena.judge.backtest import HistoryFrames, run
from tests.conftest import make_candles, make_funding

SYMS = ["BTC", "ETH", "SOL"]


class AlwaysLongBTC:
    def warmup_bars(self) -> int:
        return 24

    def decide(self, snap):
        return {"BTC": Target(weight=1.0)}


class AlwaysShortBTC:
    def warmup_bars(self) -> int:
        return 0

    def decide(self, snap):
        return {"BTC": Target(weight=-1.0)}


class Flat:
    def warmup_bars(self) -> int:
        return 0

    def decide(self, snap):
        return {}


def flat_candles(bars: int, price: float = 100.0, start="2024-01-01T01:00:00Z") -> pd.DataFrame:
    idx = pd.date_range(start, periods=bars, freq="1h", tz="UTC")
    return pd.DataFrame(
        {"symbol": "BTC", "ts": idx, "open": price, "high": price, "low": price, "close": price, "volume": 1.0}
    )


def test_always_long_btc_matches_book_compounding():
    candles = make_candles(SYMS, bars=24 * 60, drift={"BTC": 0.0005})
    fees = FeeModel()
    start, end = "2024-01-10", candles["ts"].max()
    res = run(AlwaysLongBTC(), HistoryFrames(candles), SYMS, start, end, fees)
    assert res.decisions == len(res.rows) > 0
    assert res.returns.index[0] >= pd.Timestamp(start, tz="UTC")
    assert res.turnover == pytest.approx(1.0)  # one entry, never rebalanced
    btc = candles[candles.symbol == "BTC"].set_index("ts")["close"]
    first = btc.loc[res.returns.index[0]]
    last = btc.loc[res.returns.index[-1]]
    expected = 10_000.0 * (1 - fees.perp_cost) * last / first
    assert res.nav.iloc[-1] == pytest.approx(expected, rel=1e-9)
    assert res.nav.iloc[-1] > 10_000.0


def test_flat_competitor_zero_returns():
    candles = make_candles(SYMS, bars=24 * 30)
    res = run(
        Flat(), HistoryFrames(candles, make_funding(SYMS, candles)), SYMS, "2024-01-01", candles["ts"].max(), FeeModel()
    )
    assert res.decisions == 0
    assert len(res.rows) == 24 * 30
    assert (res.returns == 0).all()
    assert (res.nav == 10_000.0).all()
    assert res.turnover == 0.0


def test_funding_accrual_short_on_flat_prices():
    candles = flat_candles(24 * 20)
    funding = make_funding(["BTC"], candles, rate=0.0001)
    res = run(
        AlwaysShortBTC(),
        HistoryFrames(candles, funding),
        ["BTC"],
        "2024-01-01",
        candles["ts"].max(),
        FeeModel(0.0, 0.0, 0.0),
    )
    first_ts = res.returns.index[0]
    stamps = funding[(funding.ts > first_ts) & (funding.ts <= candles["ts"].max())]
    assert len(stamps) > 10
    expected = (1 + 0.0001) ** len(stamps) - 1
    assert res.nav.iloc[-1] / 10_000.0 - 1 == pytest.approx(expected, rel=1e-9)
    assert sum(r.funding_pnl for r in res.rows) == pytest.approx(0.0001 * len(stamps))
    assert res.returns[res.returns != 0].size == len(stamps)


def test_warmup_skips_bars():
    candles = make_candles(["BTC"], bars=200)
    res = run(AlwaysLongBTC(), HistoryFrames(candles), ["BTC"], candles["ts"].min(), candles["ts"].max(), FeeModel())
    assert len(res.rows) == 200 - 23  # first decision when 24 bars are visible


def test_no_lookahead_when_history_extended():
    long_hist = make_candles(SYMS, bars=24 * 90, drift={"BTC": 0.001})
    end = pd.Timestamp("2024-02-15T00:00:00Z")
    short_hist = long_hist[long_hist.ts <= end]
    fund = make_funding(SYMS, long_hist)
    a = run(AlwaysLongBTC(), HistoryFrames(short_hist, fund[fund.ts <= end]), SYMS, "2024-01-10", end, FeeModel())
    b = run(AlwaysLongBTC(), HistoryFrames(long_hist, fund), SYMS, "2024-01-10", end, FeeModel())
    pd.testing.assert_series_equal(a.returns, b.returns)
    assert a.returns.index[-1] == end


def test_speed_single_symbol_2000_bars():
    candles = make_candles(["BTC"], bars=2000)
    t0 = time.perf_counter()
    run(
        AlwaysLongBTC(),
        HistoryFrames(candles, make_funding(["BTC"], candles)),
        ["BTC"],
        candles["ts"].min(),
        candles["ts"].max(),
        FeeModel(),
    )
    assert time.perf_counter() - t0 < 3.0


# --------------------------------------------------------------------------- point-in-time universe


def _pit_history(bars=24 * 60):
    """OLD trades throughout; NEW only appears halfway."""
    import numpy as np

    idx = pd.date_range("2024-01-01", periods=bars, freq="1h", tz="UTC")
    frames = []
    for sym, first in (("OLD", 0), ("NEW", bars // 2)):
        close = 100.0 * np.exp(np.cumsum(np.full(bars, 0.0001)))
        frame = pd.DataFrame(
            {"symbol": sym, "ts": idx, "open": close, "high": close, "low": close, "close": close, "volume": 1e5}
        )
        frames.append(frame.iloc[first:])
    return HistoryFrames(candles=pd.concat(frames, ignore_index=True)), idx


class _Greedy:
    """Buys everything it can see, so the snapshot's symbol list is the only limit."""

    def warmup_bars(self):
        return 2

    def decide(self, snap):
        from arena.core.types import Target

        return {s: Target(weight=0.4) for s in snap.symbols}


def test_membership_hides_symbols_the_competitor_may_not_trade():
    history, idx = _pit_history()
    schedule = {idx[0]: ["OLD"], idx[len(idx) // 2]: ["OLD", "NEW"]}
    result = run(_Greedy(), history, ["OLD", "NEW"], idx[0], idx[-1], FeeModel(), members_at=schedule)
    held = {r.ts: r.gross for r in result.rows}
    early = [g for ts, g in held.items() if ts < idx[len(idx) // 2]]
    late = [g for ts, g in held.items() if ts > idx[len(idx) // 2] + pd.Timedelta(hours=2)]
    assert max(early) == pytest.approx(0.4)  # one symbol only
    assert max(late) == pytest.approx(0.8)  # both


def test_without_a_schedule_everything_is_visible_as_before():
    history, idx = _pit_history()
    plain = run(_Greedy(), history, ["OLD", "NEW"], idx[0], idx[-1], FeeModel())
    assert max(r.gross for r in plain.rows) == pytest.approx(0.8)


def test_per_symbol_liquidity_reaches_the_book():
    from arena.core.costs import ImpactModel, SymbolLiquidity

    history, idx = _pit_history()
    fees = FeeModel(impact=ImpactModel(capacity_nav=1_000_000.0))
    thin = {idx[0]: {"OLD": SymbolLiquidity(1_000_000.0, 0.15), "NEW": SymbolLiquidity(1_000_000.0, 0.15)}}
    deep = {idx[0]: {"OLD": SymbolLiquidity(1e9, 0.15), "NEW": SymbolLiquidity(1e9, 0.15)}}
    cheap = run(_Greedy(), history, ["OLD", "NEW"], idx[0], idx[-1], fees, liquidity_at=deep)
    dear = run(_Greedy(), history, ["OLD", "NEW"], idx[0], idx[-1], fees, liquidity_at=thin)
    assert sum(r.fees for r in dear.rows) > 10 * sum(r.fees for r in cheap.rows)
