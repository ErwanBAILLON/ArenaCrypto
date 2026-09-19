import pandas as pd

from arena.competitors.news import News
from arena.core.snapshot import Snapshot
from tests.conftest import SYMBOLS, make_candles


def _news(symbol, ts_end, sent_7d_now, sent_7d_before, n_7d=8):
    idx = pd.date_range(ts_end - pd.Timedelta(hours=47), ts_end, freq="1h")
    sent = [sent_7d_before] * 24 + [sent_7d_now] * 24
    return pd.DataFrame({"symbol": symbol, "ts": idx, "sent_24h": sent, "sent_7d": sent,
                         "n_24h": 2, "n_7d": n_7d, "shock": 0})


def test_longs_positive_rising_sentiment_only(candles):
    ts = candles["ts"].max()
    news = pd.concat([
        _news("BTC", ts, 0.4, 0.1),      # positive and rising -> long
        _news("ETH", ts, 0.4, 0.6),      # positive but falling -> skip
        _news("SOL", ts, -0.3, -0.5),    # negative -> skip
    ])
    snap = Snapshot.from_long(ts, SYMBOLS, candles, news=news)
    d = News().decide(snap)
    assert set(d) == {"BTC"}
    assert 0 < d["BTC"].weight <= 0.3


def test_stale_or_sparse_news_ignored(candles):
    ts = candles["ts"].max()
    stale = _news("BTC", ts - pd.Timedelta(hours=6), 0.5, 0.1)
    sparse = _news("ETH", ts, 0.5, 0.1, n_7d=1)
    snap = Snapshot.from_long(ts, SYMBOLS, candles, news=pd.concat([stale, sparse]))
    assert News().decide(snap) == {}


def test_density_scales_size(candles):
    ts = candles["ts"].max()
    snap_lo = Snapshot.from_long(ts, SYMBOLS, candles, news=_news("BTC", ts, 0.4, 0.1, n_7d=5))
    snap_hi = Snapshot.from_long(ts, SYMBOLS, candles, news=_news("BTC", ts, 0.4, 0.1, n_7d=20))
    assert News().decide(snap_lo)["BTC"].weight < News().decide(snap_hi)["BTC"].weight
