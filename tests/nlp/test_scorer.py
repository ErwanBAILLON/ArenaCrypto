from datetime import datetime, timezone

import pytest

from arena.core.types import Article
from arena.nlp.lexicon import classify_event, detect_assets
from arena.nlp.scorer import VaderScorer

NOW = datetime(2024, 1, 10, tzinfo=timezone.utc)


def art(title: str, summary: str = "", id: int = 1) -> Article:
    return Article(source="s", url="u", title=title, summary=summary, published_at=NOW, fetched_at=NOW, id=id)


@pytest.fixture(scope="module")
def scorer() -> VaderScorer:
    return VaderScorer(version=1)


def test_etf_approval_is_positive_btc_etf(scorer):
    (s,) = scorer.score(art("Bitcoin ETF approved by SEC"))
    assert s.asset == "BTC" and s.event_type == "etf" and s.scorer_version == 1 and s.article_id == 1
    assert s.sentiment > 0.3
    assert s.intensity == pytest.approx(min(1.0, abs(s.sentiment) * 1.5))


def test_hack_is_negative_high_intensity(scorer):
    (s,) = scorer.score(art("Solana DeFi protocol hacked, $50M drained"))
    assert s.asset == "SOL" and s.event_type == "hack"
    assert s.sentiment < -0.5 and s.intensity >= 0.8


def test_market_fallback_and_macro(scorer):
    (s,) = scorer.score(art("Markets quiet ahead of Fed"))
    assert s.asset == "MARKET" and s.event_type == "macro"


def test_ambiguous_short_aliases_require_long_form_or_cashtag():
    assert detect_assets("Optimism grows among traders") == []
    assert detect_assets("A link to the report") == []
    assert detect_assets("near the highs") == []
    assert detect_assets("$OP and OP token rally") == ["OP"]
    assert detect_assets("Near Protocol and Sui Network ship upgrades") == ["NEAR", "SUI"]
    assert detect_assets("Chainlink integrates with Arbitrum") == ["LINK", "ARB"]
    assert detect_assets("Polkadot dot com") == ["DOT"]


def test_optimism_headline_scores_market_not_op(scorer):
    scores = scorer.score(art("Optimism grows among traders"))
    assert [s.asset for s in scores] == ["MARKET"]


def test_multiple_assets_one_row_each(scorer):
    scores = scorer.score(art("Ethereum and $BTC rally", "ether inflows hit a record high"))
    assert [s.asset for s in scores] == ["BTC", "ETH"]
    assert len({(s.sentiment, s.event_type, s.intensity) for s in scores}) == 1
    assert scores[0].sentiment > 0


def test_multiword_lexicon_phrases_apply(scorer):
    (neg,) = scorer.score(art("Token suffers rug pull"))
    (pos,) = scorer.score(art("Token hits all-time high"))
    assert neg.sentiment < 0 < pos.sentiment


def test_event_rule_order():
    assert classify_event("exchange hacked after court ruling") == "hack"
    assert classify_event("SEC sues exchange") == "regulation"
    assert classify_event("new listing on exchange") == "listing"
    assert classify_event("mainnet upgrade shipped") == "protocol"
    assert classify_event("nothing happened") == "other"


def test_sentiment_and_intensity_bounded(scorer):
    for s in scorer.score(art("hack exploit drained fraud bankruptcy depeg", "rug pull rugged banned")):
        assert -1.0 <= s.sentiment <= 1.0 and 0.0 <= s.intensity <= 1.0


def test_deterministic(scorer):
    a = art("Solana DeFi protocol hacked, $50M drained", "Users report losses; team halts bridge.")
    assert scorer.score(a) == scorer.score(a)
    assert scorer.score(a) == VaderScorer(version=1).score(a)


def test_requires_persisted_article(scorer):
    with pytest.raises(ValueError):
        scorer.score(Article("s", "u", "t", "", NOW, NOW))
