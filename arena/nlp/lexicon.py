"""Static vocabularies for the rule-based news scorer.

``ASSET_ALIASES`` maps universe symbols to the phrases that name them;
``CRYPTO_LEXICON`` overlays VADER's lexicon (scale -4..4) with domain words;
``EVENT_RULES`` classify an article into an event type, first match wins.
"""

from __future__ import annotations

import re

ASSET_ALIASES: dict[str, list[str]] = {
    "BTC": ["bitcoin", "btc", "$btc"],
    "ETH": ["ethereum", "ether", "eth", "$eth"],
    "SOL": ["solana", "sol", "$sol"],
    "BNB": ["bnb", "binance coin", "$bnb"],
    "XRP": ["xrp", "ripple", "$xrp"],
    "DOGE": ["dogecoin", "doge", "$doge"],
    "ADA": ["cardano", "ada", "$ada"],
    "AVAX": ["avalanche", "avax", "$avax"],
    "LINK": ["chainlink", "link token", "$link"],
    "DOT": ["polkadot", "$dot"],
    "LTC": ["litecoin", "ltc", "$ltc"],
    "NEAR": ["near protocol", "$near"],
    "SUI": ["sui network", "$sui"],
    "ARB": ["arbitrum", "arb", "$arb"],
    "OP": ["op token", "$op"],
}

CRYPTO_LEXICON: dict[str, float] = {
    "hack": -3.0,
    "hacked": -3.0,
    "exploit": -3.0,
    "drained": -3.0,
    "rug pull": -3.5,
    "rugged": -3.0,
    "delist": -2.5,
    "delisted": -2.5,
    "ban": -2.0,
    "banned": -2.0,
    "lawsuit": -2.0,
    "sues": -2.5,
    "charged": -2.0,
    "fraud": -3.0,
    "bankruptcy": -3.5,
    "insolvent": -3.0,
    "liquidation": -1.5,
    "liquidations": -1.5,
    "outage": -1.5,
    "halt": -1.5,
    "depeg": -3.0,
    "etf approved": 3.0,
    "approval": 2.0,
    "approved": 2.0,
    "listing": 1.5,
    "listed": 1.5,
    "upgrade": 1.0,
    "mainnet": 1.0,
    "partnership": 1.0,
    "adoption": 1.5,
    "all-time high": 2.0,
    "ath": 1.5,
    "record high": 2.0,
    "inflow": 1.5,
    "inflows": 1.5,
    "outflow": -1.5,
    "outflows": -1.5,
    "halving": 1.0,
    "buyback": 1.5,
}

# First match wins. ``etf`` precedes ``regulation`` because ETF headlines
# almost always name the SEC and would otherwise never be classified as etf.
# Keywords match whole words (a prefix match on "sec" would hit "second"),
# so common inflections are listed explicitly.
EVENT_RULES: list[tuple[str, list[str]]] = [
    ("hack", ["hack", "hacked", "hacker", "hacks", "exploit", "exploited", "drained", "stolen", "rug", "rugged"]),
    ("etf", ["etf", "etfs"]),
    (
        "regulation",
        [
            "sec",
            "cftc",
            "regulator",
            "regulators",
            "regulation",
            "lawsuit",
            "sues",
            "sued",
            "ban",
            "banned",
            "law",
            "mica",
            "court",
        ],
    ),
    ("listing", ["listing", "listed", "delist", "delisted", "delisting"]),
    ("macro", ["fed", "fomc", "cpi", "inflation", "rates", "treasury", "tariff", "tariffs"]),
    ("protocol", ["upgrade", "mainnet", "hard fork", "testnet", "airdrop"]),
]

DEFAULT_EVENT = "other"


def phrase_pattern(phrase: str) -> re.Pattern[str]:
    """Case-insensitive whole-word matcher for a phrase (``$`` cashtags included)."""
    body = re.escape(phrase)
    lead = r"(?<![\w$])" if phrase.startswith("$") else r"\b"
    return re.compile(lead + body + r"\b", re.IGNORECASE)


ASSET_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    symbol: [phrase_pattern(a) for a in aliases] for symbol, aliases in ASSET_ALIASES.items()
}

EVENT_PATTERNS: list[tuple[str, list[re.Pattern[str]]]] = [
    (event, [phrase_pattern(k) for k in keywords]) for event, keywords in EVENT_RULES
]


def detect_assets(text: str) -> list[str]:
    """Universe symbols named in ``text``, in universe order."""
    return [symbol for symbol, patterns in ASSET_PATTERNS.items() if any(p.search(text) for p in patterns)]


def classify_event(text: str) -> str:
    """Event type of ``text`` by the first matching rule, else ``"other"``."""
    for event, patterns in EVENT_PATTERNS:
        if any(p.search(text) for p in patterns):
            return event
    return DEFAULT_EVENT
