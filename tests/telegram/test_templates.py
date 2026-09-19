from arena.telegram.templates import MAX_LEN, daily_digest, event_alert, signal_alert


def test_signal_long_wording():
    s = signal_alert("trend_ts v3", "ETH", 0.35, 0.71, "perp", "bull_vol", 0.00012, {"ema50>ema200": True, "r30": 0.18})
    assert s.startswith("⚔️ trend_ts v3 → LONG ETH 0.35 (conv 0.71)")
    assert "regime bull_vol" in s
    assert "fund +0.012%/8h" in s
    assert "why: ema50>ema200=true, r30=0.18" in s


def test_signal_short_carry_flat():
    assert "SHORT BTC 0.40" in signal_alert("x", "BTC", -0.4, 0.5, "perp", None, None, {})
    assert "CARRY SOL 0.20" in signal_alert("x", "SOL", 0.2, 0.5, "carry", None, None, {})
    assert "FLAT SOL 0.00" in signal_alert("x", "SOL", 0.0, 0.5, "perp", None, None, {})
    assert "FLAT SOL 0.00" in signal_alert("x", "SOL", 0.0, 0.5, "carry", None, None, {})


def test_signal_no_regime_no_funding():
    s = signal_alert("x", "BTC", 0.1, 0.5, "perp", None, None, {})
    assert "regime n/a" in s
    assert "fund" not in s
    assert s.endswith("why: -")


def test_reason_truncated_to_six_pairs_and_3_sig_digits():
    reason = {f"k{i}": 1.23456 * (i + 1) for i in range(10)}
    s = signal_alert("x", "BTC", 0.5, 0.5, "perp", None, None, reason)
    why = s.split("why: ")[1]
    pairs = why.split(", ")
    assert len(pairs) == 6
    assert pairs[0] == "k0=1.23"
    assert "k6" not in why


def _digest():
    lb = [
        {
            "name": "trend_ts",
            "status": "champion",
            "role": "competitor",
            "sharpe_30d": 1.52,
            "ret_30d": 0.081,
            "mdd_30d": -0.042,
            "nav": 10810.0,
        },
        {
            "name": "xs_mom",
            "status": "challenger",
            "role": "competitor",
            "sharpe_30d": -0.3,
            "ret_30d": -0.012,
            "mdd_30d": -0.09,
            "nav": 9880.0,
        },
        {
            "name": "null_random",
            "status": "champion",
            "role": "null",
            "sharpe_30d": 0.1,
            "ret_30d": 0.0,
            "mdd_30d": -0.05,
            "nav": 10000.0,
        },
    ]
    return daily_digest(
        "2026-09-19",
        lb,
        [("trend_ts", 0.7), ("carry", 0.3)],
        1.23,
        [{"name": "xs_mom", "days_in_arena": 31, "decisions": 58}],
        ["trend_ts: live Sharpe 0.4 vs backtest 1.8"],
    )


def test_digest_sections_present():
    d = _digest()
    for section in (
        "📊 Arena digest 2026-09-19",
        "Leaderboard (30d)",
        "Allocator opinion",
        "Null 95th pct Sharpe: 1.23",
        "Challengers",
        "Drift",
    ):
        assert section in d
    assert "🏆 trend_ts" in d
    assert "🧪 xs_mom" in d
    assert "⚪ null_random" in d
    assert "trend_ts → 70%" in d
    assert "31/42d, 58/100 dec" in d
    assert "• trend_ts: live Sharpe" in d
    assert "+8.1%" in d


def test_digest_empty_inputs():
    d = daily_digest("2026-01-01", [], [], None, [], [])
    assert "Null 95th pct Sharpe: n/a" in d
    assert d.count("none") == 3
    assert "(empty)" in d


def test_length_cap():
    drift = [f"alert number {i} " + "x" * 80 for i in range(200)]
    d = daily_digest("2026-01-01", [], [], None, [], drift)
    assert len(d) <= MAX_LEN
    assert d.endswith("…")
    e = event_alert("error", "y" * 5000)
    assert len(e) <= MAX_LEN and e.endswith("…")


def test_event_alert_emojis():
    assert event_alert("promotion", "x").startswith("🏆 x")
    assert event_alert("rejected", "x").startswith("🚫 x")
    assert event_alert("stale", "x").startswith("⏳ x")
    assert event_alert("error", "x").startswith("❗ x")
    assert event_alert("info", "x").startswith("ℹ️ x")
    assert event_alert("unknown", "x").startswith("ℹ️ x")
