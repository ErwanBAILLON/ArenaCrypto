"""Deterministic VADER-based news scorer with a crypto overlay (scorer version 1)."""

from __future__ import annotations

import re

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from arena.core.types import Article, ArticleScore
from arena.nlp.lexicon import CRYPTO_LEXICON, classify_event, detect_assets, phrase_pattern

MARKET = "MARKET"

TITLE_WEIGHT = 0.6
BODY_WEIGHT = 0.4
HIGH_IMPACT_EVENTS = frozenset({"hack", "regulation", "etf"})
HIGH_IMPACT_BOOST = 1.5


def _token(phrase: str) -> str:
    """Single-token form of a lexicon phrase so VADER can look it up."""
    return phrase.replace(" ", "_").replace("-", "_")


class VaderScorer:
    """Scores an article into one ``ArticleScore`` per detected asset (``MARKET`` when none).

    Sentiment = 0.6 * compound(title) + 0.4 * compound(title + summary), with
    the crypto lexicon merged into VADER's. Intensity = |sentiment|, boosted
    1.5x (capped at 1) for hack / regulation / etf events.
    """

    def __init__(self, version: int = 1) -> None:
        self.version = version
        self._analyzer = SentimentIntensityAnalyzer()
        self._analyzer.lexicon.update({_token(k): v for k, v in CRYPTO_LEXICON.items()})
        self._phrases: list[tuple[re.Pattern[str], str]] = [
            (phrase_pattern(k), _token(k)) for k in CRYPTO_LEXICON if _token(k) != k
        ]

    def _normalise(self, text: str) -> str:
        """Collapse multi-word lexicon phrases into single tokens."""
        for pattern, token in self._phrases:
            text = pattern.sub(token, text)
        return text

    def _compound(self, text: str) -> float:
        return float(self._analyzer.polarity_scores(self._normalise(text))["compound"])

    def score(self, article: Article) -> list[ArticleScore]:
        """Score one article; requires ``article.id`` to be set."""
        if article.id is None:
            raise ValueError("article must be persisted (id is None)")
        text = f"{article.title}. {article.summary}"
        sentiment = TITLE_WEIGHT * self._compound(article.title) + BODY_WEIGHT * self._compound(text)
        sentiment = max(-1.0, min(1.0, sentiment))
        event_type = classify_event(text)
        boost = HIGH_IMPACT_BOOST if event_type in HIGH_IMPACT_EVENTS else 1.0
        intensity = min(1.0, abs(sentiment) * boost)
        assets = detect_assets(text) or [MARKET]
        return [
            ArticleScore(
                article_id=article.id,
                asset=asset,
                sentiment=sentiment,
                event_type=event_type,
                intensity=intensity,
                scorer_version=self.version,
            )
            for asset in assets
        ]
