from arena.core.universe import load_universe


def test_default_universe_loads():
    u = load_universe()
    assert len(u.symbols) == 15
    assert "BTC" in u.symbols and "OP" in u.symbols
    assert u.fees.perp_taker == 0.0005
    assert u.binance_symbol("BTC") == "BTCUSDT"
    assert u.history_start.year == 2024


def test_a_membership_universe_names_its_reference(tmp_path):
    from arena.core.universe import load_universe

    p = tmp_path / "wide.yaml"
    p.write_text("name: wide\nexchange: binance\nbinance_suffix: ''\nreference: BTCUSDT\nmembership: {enabled: true}\n")
    u = load_universe(p)
    assert u.symbols == [] and u.reference == "BTCUSDT"
    p.write_text("name: w2\nexchange: binance\nbinance_suffix: USDT\n")
    assert load_universe(p).reference == "BTCUSDT"  # nothing configured: the exchange's BTC perp
