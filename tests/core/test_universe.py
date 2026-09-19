from arena.core.universe import load_universe


def test_default_universe_loads():
    u = load_universe()
    assert len(u.symbols) == 15
    assert "BTC" in u.symbols and "OP" in u.symbols
    assert u.fees.perp_taker == 0.0005
    assert u.binance_symbol("BTC") == "BTCUSDT"
    assert u.history_start.year == 2024
