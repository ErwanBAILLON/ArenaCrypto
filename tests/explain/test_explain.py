from arena.core.types import Target
from arena.explain.families import CARDS, card, params_in_words
from arena.explain.reasons import explain_decision, explain_target
from arena.explain.verdicts import explain_gate, explain_status, money


def test_every_registered_family_has_a_card():
    from arena.competitors.base import REGISTRY

    for fam in REGISTRY:
        assert fam in CARDS, fam
        assert card(fam).inputs


def test_reasons_are_sentences_with_values():
    t = Target(0.2, kind="carry", reason={"mean_funding_8h": 0.00012, "rank": 0})
    s = explain_target("carry", "SUI", t)
    assert "SUI" in s and "0.0120 %" in s and "rang 1" in s
    t2 = Target(-0.3, reason={"r_short": -0.15, "r_long": -0.3, "vol": 0.6})
    s2 = explain_target("trend_ts", "SOL", t2)
    assert s2.startswith("Vente SOL") and "-15.0 %" in s2
    assert "Sort de BTC" in explain_target("trend_ts", "BTC", Target(0.0, reason={"exit": True}))
    lines = explain_decision(
        "xs_momentum",
        {"A": Target(0.1, reason={"rank": 0, "score": 0.2}), "B": Target(-0.5, reason={"rank": 14, "score": -0.3})},
    )
    assert lines[0].startswith("Vente B")


def test_params_in_words_and_gate():
    assert any("14 jours" in s for s in params_in_words("carry", {}))
    m = {
        "folds_positive_frac": 0.75,
        "fold_sharpes": [1, 2, -1, 3],
        "sharpe": 2.1,
        "null_threshold": 0.8,
        "dsr": 0.97,
        "bootstrap_p": 0.02,
        "max_drawdown": 0.12,
        "decisions": 500,
        "n_trials": 7,
        "total_return": 0.31,
    }
    lines = explain_gate(m, [])
    assert lines[0].startswith("✓ A gagné dans 3 périodes de test sur 4")
    assert all(line.startswith("✓") for line in lines[:-1])
    bad = explain_gate({**m, "sharpe": 0.1}, ["sharpe_above_null"])
    assert any(line.startswith("✗") and "la chance" in line for line in bad)
    assert "Prétendant" in explain_status("challenger", "competitor", 10, 5)
    assert money(0.0123) == "+123 €"
