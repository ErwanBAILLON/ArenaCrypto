from datetime import UTC, datetime

from arena.telegram.templates import (
    MAX_LEN,
    ChallengerView,
    ChampionView,
    DigestContext,
    PositionView,
    daily_digest,
    event_alert,
    signal_alert,
    split_name,
)


def test_split_name():
    assert split_name("carry_v1") == ("carry", 1)
    assert split_name("trend_ts_v12") == ("trend_ts", 12)
    assert split_name("null_random_0") == ("null_random_0", None)


def test_signal_carry_wording_from_name():
    s = signal_alert(
        "carry_v1",
        "SUI",
        0.2,
        0.8,
        "carry",
        "bull_calm",
        0.00012,
        {"mean_funding_8h": 0.00012, "rank": 0},
        status="champion",
    )
    lines = s.split("\n")
    assert lines[0] == "⚔️ Carry de funding (carry v1, champion)"
    assert "Carry sur SUI" in lines[1]
    assert "20 % du capital" in lines[1]
    assert "funding moyen 0,0120 % par 8 h" in lines[1]
    assert "rang 1 des mieux payés" in lines[1]
    assert lines[2] == "Marché : haussier calme. Capital virtuel : 10 000 €."


def test_signal_explicit_family_kwargs_win_over_name():
    s = signal_alert(
        "weird-name",
        "ETH",
        0.35,
        0.7,
        "perp",
        "bull_vol",
        None,
        {"r_short": 0.18, "r_long": 0.4},
        family="trend_ts",
        version=3,
    )
    assert s.startswith("⚔️ Tendance (momentum temporel) (trend_ts v3)")
    assert "Achat ETH (35 % du capital)" in s
    assert "monte depuis 30 jours (+18,0 %)" in s
    assert "Marché : haussier volatil." in s


def test_signal_short_and_exit():
    s = signal_alert("xs_momentum_v2", "BTC", -0.4, 0.5, "perp", None, None, {"rank": 4, "score": -0.12})
    assert "Vente BTC (40 % du capital)" in s
    assert "parmi les plus faibles" in s
    assert "Marché : indéterminé." in s
    flat = signal_alert("trend_ts_v1", "SOL", 0.0, 0.5, "perp", "range", 0.0001, {})
    assert "Sort de SOL : les conditions d'entrée ne tiennent plus." in flat
    assert "Funding actuel sur SOL : 0,0100 % par 8 h" in flat


def test_signal_unknown_family_still_speaks():
    s = signal_alert("mystery_v1", "BTC", 0.1, 0.5, "perp", None, None, {})
    assert s.startswith("⚔️ mystery (mystery v1)")
    assert "Achat BTC (10 % du capital)." in s


def _ctx() -> DigestContext:
    return DigestContext(
        date_str="20/09/2026",
        champions=[
            ChampionView(
                "carry_v1",
                "carry",
                1,
                [PositionView("SUI", 0.2, 0.8, "carry", {"mean_funding_8h": 0.00012, "rank": 0})],
                pnl_1d=0.0021,
                pnl_7d=0.011,
                pnl_30d=0.081,
                btc_30d=0.05,
            ),
            ChampionView("trend_ts_v3", "trend_ts", 3, [], pnl_1d=None, pnl_7d=-0.004, pnl_30d=-0.012, btc_30d=0.05),
        ],
        changes=["carry v1 entre sur SUI (20 % du capital)."],
        challengers=[
            ChallengerView(
                "xs_momentum_v2", "xs_momentum", 2, 31, 58, pnl_30d=-0.012, sharpe_30d=0.4, champion_sharpe_30d=1.5
            ),
            ChallengerView("news_v1", "news", 1, 5, 3, pnl_30d=0.03, sharpe_30d=2.0, champion_sharpe_30d=None),
        ],
        btc_30d_eur=500.0,
        null95=1.23,
        last_tick=datetime(2026, 9, 20, 6, 4, tzinfo=UTC),
        evaluated=9,
        drift_lines=["trend_ts_v3 perd de l'argent en direct : Sharpe 30 j -1,20, rendement -12,3%."],
    )


def test_digest_four_sections_in_french():
    d = daily_digest(_ctx())
    for section in (
        "📊 L'arène ce matin — 20/09/2026",
        "Qui parle :",
        "Ce qui a changé depuis hier :",
        "Les prétendants :",
        "Repères : garder du BTC 30 j : +500 € ; la chance (95e centile) : Sharpe 1,23.",
        "Santé : dernier passage 06:04 UTC, 9 modèles évalués, dérives :",
    ):
        assert section in d, section
    assert "• Carry de funding (carry v1)" in d
    assert "Carry sur SUI (20 % du capital)" in d
    assert "P&L hier +21 € / 7 j +110 € / 30 j +810 € ; devant « garder du BTC » de 310 €." in d
    assert "• Tendance (momentum temporel) (trend_ts v3)" in d
    assert "À plat : aucune position, il attend un signal." in d
    assert "P&L hier n/d / 7 j -40 € / 30 j -120 € ; derrière « garder du BTC » de 620 €." in d
    assert "• carry v1 entre sur SUI (20 % du capital)." in d
    assert "• xs_momentum v2 (Momentum relatif (classement)) : 31/42 jours, 58/100 décisions." in d
    assert "30 j : -120 € ; en retard sur le champion ; en dessous de la chance." in d
    assert "30 j : +300 € ; pas de champion à battre dans sa famille ; au-dessus de la chance." in d
    assert "• trend_ts_v3 perd de l'argent en direct" in d
    assert "Sharpe" not in d.split("Les prétendants :")[1].split("Repères")[0]  # in words, not numbers


def test_digest_empty_inputs():
    d = daily_digest(DigestContext(date_str="01/01/2026"))
    assert "• Personne : aucun champion en lice." in d
    assert "Ce qui a changé depuis hier :\n• rien" in d
    assert "Les prétendants :\n• aucun" in d
    assert "Repères : garder du BTC 30 j : n/d ; la chance (95e centile) : n/d." in d
    assert d.endswith("Santé : aucun passage enregistré, dérives : aucune")


def test_length_cap():
    drift = [f"alerte numéro {i} " + "x" * 80 for i in range(200)]
    d = daily_digest(DigestContext(date_str="01/01/2026", drift_lines=drift))
    assert len(d) <= MAX_LEN
    assert d.endswith("…")
    e = event_alert("error", "y" * 5000)
    assert len(e) <= MAX_LEN and e.endswith("…")


def test_event_alert_labels_and_emojis():
    assert event_alert("stale", "x").startswith("⏳ Données en retard : x")
    assert event_alert("error", "x").startswith("❗ Erreur : x")
    assert event_alert("drift", "x").startswith("📉 Dérive : x")
    assert event_alert("info", "x") == "ℹ️ x"
    assert event_alert("unknown", "x") == "ℹ️ x"


def test_event_alert_translates_runner_details():
    assert event_alert("stale", "last closed bar 2026-09-19 06:00 UTC, lag 3.2h") == (
        "⏳ Données en retard : dernière bougie close 2026-09-19 06:00 UTC, 3,2 h de retard."
    )
    assert event_alert("stale", "no candles at all") == (
        "⏳ Données en retard : aucune bougie en base, le marché n'a jamais été ingéré."
    )
    assert event_alert("drift", "carry_v1: 30d Sharpe -1.20, return -12.3%") == (
        "📉 Dérive : carry_v1 perd de l'argent en direct : Sharpe 30 j -1,20, rendement -12,3%."
    )
    assert event_alert("drift", "news_v1: no non-zero target for 7+ days") == (
        "📉 Dérive : news_v1 n'a pris aucune position depuis 7 jours ou plus."
    )
    assert event_alert("error", "carry_v1: nav is nan") == (
        "❗ Erreur : carry_v1 : capital virtuel invalide (nan), livre cassé."
    )
    assert (
        event_alert("info", "new challenger carry_v2: optimized k=3") == "ℹ️ Nouveau prétendant carry_v2 : optimized k=3"
    )


def test_promotion_with_payload():
    e = event_alert(
        "promotion",
        "carry_v2 promoted to champion of carry (was carry_v1)",
        {
            "challenger": "carry_v2",
            "family": "carry",
            "old_champion": "carry_v1",
            "challenger_sharpe": 1.4,
            "champion_sharpe": 0.9,
            "null95": 0.7,
        },
    )
    assert e == (
        "🏆 Promotion : carry_v2 devient champion de Carry de funding (était carry_v1). "
        "Raison : Sharpe en direct 1,40 contre 0,90, au-dessus de la chance (0,70)."
    )
    first = event_alert("promotion", "x", {"challenger": "news_v1", "family": "news", "challenger_sharpe": 2.0})
    assert (
        "devient champion de Actualité (sentiment lent) (la famille n'en avait pas). Raison : Sharpe en direct 2,00."
        in first
    )
    assert event_alert("promotion", "raw text") == "🏆 Promotion : raw text"


def test_rejected_failed_list_translated():
    e = event_alert("rejected", "carry_v1 rejected at gate (dsr, max_drawdown); enters as challenger")
    assert e.startswith("🚫 Refusé à l'entrée : carry_v1 (critères manqués : ")
    assert "un Sharpe déflaté crédible à 90 % vu le nombre d'essais ; un pire creux sous 30 %" in e
    assert e.endswith("Il entre quand même comme prétendant, jugé en direct.")
    e2 = event_alert("rejected", "trend_ts_v2 rejected", {"failed": ["min_decisions"]})
    assert e2 == "🚫 Refusé à l'entrée : trend_ts_v2 (critères manqués : au moins 30 décisions)."


def test_rejected_with_metrics_uses_explain_gate():
    e = event_alert(
        "rejected",
        "carry_v1 rejected at gate (max_drawdown, dsr)",
        {
            "failed": ["max_drawdown", "dsr"],
            "metrics": {"sharpe": 1.1, "null_threshold": 0.8, "dsr": 0.62, "n_trials": 12, "max_drawdown": 0.41},
        },
    )
    assert e.startswith("🚫 Refusé à l'entrée : carry_v1\n")
    assert "✗ Pire creux depuis un sommet : 41,0 % (limite 30 %)." in e
    assert "✗ Probabilité que ce résultat ne soit pas dû au nombre d'essais : 62 %" in e
    assert "✓" not in e  # only the missed criteria are shown
