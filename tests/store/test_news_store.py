from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from arena.core.types import Article, ArticleScore
from arena.store import news as repo

T0 = datetime(2024, 3, 1, 12, tzinfo=timezone.utc)
V = 1


def art(title: str, published_at: datetime, source: str = "feed", url: str | None = None) -> Article:
    return Article(source=source, url=url or f"https://x/{title}", title=title, summary="",
                   published_at=published_at, fetched_at=published_at)


def score(article_id: int, asset: str, sentiment: float, intensity: float = 0.3, version: int = V) -> ArticleScore:
    return ArticleScore(article_id=article_id, asset=asset, sentiment=sentiment, event_type="other",
                        intensity=intensity, scorer_version=version)


def test_upsert_articles_returns_only_new_ids(conn):
    a1, a2 = art("one", T0), art("two", T0)
    ids = repo.upsert_articles(conn, [a1, a2])
    assert len(ids) == 2
    assert repo.upsert_articles(conn, [a1, art("three", T0)]) != ids
    assert len(repo.upsert_articles(conn, [a1, a2])) == 0


def test_unscored_articles_and_upsert_scores(conn):
    ids = repo.upsert_articles(conn, [art("a", T0), art("b", T0 + timedelta(hours=1))])
    pending = repo.unscored_articles(conn, V)
    assert [p.id for p in pending] == ids
    assert pending[0].title == "a" and pending[0].published_at.tzinfo is not None

    repo.upsert_scores(conn, [score(ids[0], "BTC", 0.5)])
    assert [p.id for p in repo.unscored_articles(conn, V)] == [ids[1]]
    assert [p.id for p in repo.unscored_articles(conn, V + 1)] == ids  # other version untouched

    repo.upsert_scores(conn, [score(ids[0], "BTC", -0.5)])  # overwrite same key
    with conn.cursor() as cur:
        cur.execute("SELECT sentiment FROM article_scores WHERE article_id = %s", (ids[0],))
        assert cur.fetchone()["sentiment"] == -0.5
    assert repo.unscored_articles(conn, V, limit=1) == [pending[1]]


def test_macro_events_round_trip(conn):
    df = pd.DataFrame({
        "ts": [T0, T0 + timedelta(days=1)], "currency": ["USD", "USD"],
        "title": ["CPI", "FOMC"], "impact": ["high", "high"],
    })
    assert repo.upsert_macro_events(conn, df) == 2
    assert repo.upsert_macro_events(conn, df) == 0
    idx = repo.macro_event_times(conn, T0, T0 + timedelta(hours=1))
    assert isinstance(idx, pd.DatetimeIndex) and str(idx.tz) == "UTC"
    assert list(idx) == [pd.Timestamp(T0)]
    assert len(repo.macro_event_times(conn, T0 - timedelta(days=5), T0 - timedelta(days=4))) == 0


@pytest.fixture
def seeded(conn):
    """Three scored articles: BTC at T0-2h (0.8, shock), BTC at T0-30h (-0.4), MARKET at T0-1h (0.2)."""
    ids = repo.upsert_articles(conn, [
        art("btc recent", T0 - timedelta(hours=2)),
        art("btc old", T0 - timedelta(hours=30)),
        art("market", T0 - timedelta(hours=1)),
    ])
    repo.upsert_scores(conn, [
        score(ids[0], "BTC", 0.8, intensity=0.9),
        score(ids[1], "BTC", -0.4, intensity=0.2),
        score(ids[2], "MARKET", 0.2, intensity=0.1),
        score(ids[0], "BTC", 0.0, intensity=0.0, version=V + 1),  # other version must be ignored
    ])
    return ids


def test_news_features_grid_and_math(conn, seeded):
    start, end = T0 - timedelta(hours=3), T0 + timedelta(hours=2)
    df = repo.news_features(conn, ["BTC", "ETH"], start, end, V)

    assert list(df.columns) == ["symbol", "ts", "sent_24h", "sent_7d", "n_24h", "n_7d", "shock"]
    assert set(df["symbol"]) == {"BTC", "ETH", "MARKET"}
    assert str(df["ts"].dt.tz) == "UTC"
    assert (df.groupby("symbol").size() == 6).all()  # hourly grid inclusive of both ends

    btc = df[df.symbol == "BTC"].set_index("ts")
    at = btc.loc[pd.Timestamp(T0)]
    assert at["n_24h"] == 1 and at["sent_24h"] == pytest.approx(0.8)
    assert at["n_7d"] == 2 and at["sent_7d"] == pytest.approx(0.2)  # mean(0.8, -0.4)
    assert at["shock"] == 1

    before = btc.loc[pd.Timestamp(T0 - timedelta(hours=3))]  # nothing published yet in the 24h window
    assert before["n_24h"] == 0 and before["sent_24h"] == 0.0 and before["shock"] == 0
    assert before["n_7d"] == 1 and before["sent_7d"] == pytest.approx(-0.4)

    eth = df[df.symbol == "ETH"]
    assert (eth[["sent_24h", "sent_7d", "n_24h", "n_7d", "shock"]] == 0).all().all()

    mkt = df[df.symbol == "MARKET"].set_index("ts").loc[pd.Timestamp(T0)]
    assert mkt["n_24h"] == 1 and mkt["sent_24h"] == pytest.approx(0.2) and mkt["shock"] == 0


def test_news_features_window_is_right_closed(conn, seeded):
    # BTC article at T0-2h: included at h = T0-2h (published_at <= h), excluded at h = T0-3h.
    df = repo.news_features(conn, ["BTC"], T0 - timedelta(hours=3), T0 - timedelta(hours=2), V)
    btc = df[df.symbol == "BTC"].set_index("ts")
    assert btc.loc[pd.Timestamp(T0 - timedelta(hours=3)), "n_24h"] == 0
    assert btc.loc[pd.Timestamp(T0 - timedelta(hours=2)), "n_24h"] == 1
    # 24h later the article drops out of the 24h window but stays in the 7d one.
    later = repo.news_features(conn, ["BTC"], T0 + timedelta(hours=22), T0 + timedelta(hours=22), V)
    row = later[later.symbol == "BTC"].iloc[0]
    assert row["n_24h"] == 0 and row["n_7d"] == 2 and row["shock"] == 0
