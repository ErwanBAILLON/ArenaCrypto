"""Gate metrics and promotion state in plain French."""

from __future__ import annotations

from typing import Any


def explain_gate(metrics: dict[str, Any], failed: list[str] | None = None) -> list[str]:
    """One sentence per criterion, with the number behind it."""
    failed = set(failed or [])
    m = metrics
    lines: list[str] = []
    fp = m.get("folds_positive_frac")
    if fp is not None:
        n = len(m.get("fold_sharpes", [])) or 0
        won = round(float(fp) * n) if n else None
        txt = (
            f"A gagné dans {won} périodes de test sur {n}"
            if n
            else f"A gagné dans {float(fp) * 100:.0f} % des périodes de test"
        )
        lines.append(("✗ " if "folds_positive" in failed else "✓ ") + txt + " (il en faut deux sur trois).")
    if "sharpe" in m and "null_threshold" in m:
        ok = "sharpe_above_null" not in failed
        lines.append(
            ("✓ " if ok else "✗ ")
            + f"Rendement ajusté du risque {float(m['sharpe']):.2f} contre {float(m['null_threshold']):.2f} pour le 95e centile des modèles aléatoires"
            + (" : il bat la chance." if ok else " : la chance fait aussi bien.")
        )
    if "dsr" in m:
        ok = "dsr" not in failed
        lines.append(
            ("✓ " if ok else "✗ ")
            + f"Probabilité que ce résultat ne soit pas dû au nombre d'essais : {float(m['dsr']) * 100:.0f} % "
            f"({int(m.get('n_trials', 0))} essais comptés dans cette famille ; il en faut 90 %)."
        )
    if "bootstrap_p" in m:
        ok = "bootstrap_p" not in failed
        lines.append(
            ("✓ " if ok else "✗ ")
            + f"Probabilité d'obtenir ça par hasard en mélangeant ses propres rendements : {float(m['bootstrap_p']) * 100:.0f} % (il faut moins de 10 %)."
        )
    if "max_drawdown" in m:
        ok = "max_drawdown" not in failed
        lines.append(
            ("✓ " if ok else "✗ ")
            + f"Pire creux depuis un sommet : {float(m['max_drawdown']) * 100:.1f} % (limite 30 %)."
        )
    if "decisions" in m:
        ok = "min_decisions" not in failed
        lines.append(
            ("✓ " if ok else "✗ ") + f"{int(m['decisions'])} décisions prises (il en faut au moins 30 pour juger)."
        )
    if "total_return" in m:
        lines.append(f"Résultat net de frais sur toute la période de test : {float(m['total_return']) * 100:+.1f} %.")
    return lines


def explain_status(
    status: str,
    role: str,
    days_in_arena: int | None,
    decisions: int | None,
    min_days: int = 42,
    min_decisions: int = 100,
) -> str:
    if role != "competitor":
        return "Repère : toujours présent, jamais promu ni retiré."
    if status == "champion":
        return "Champion : c'est lui qui parle. Il reste champion tant qu'un prétendant ne fait pas mieux que lui en direct."
    if status == "challenger":
        d = days_in_arena or 0
        n = decisions or 0
        need = []
        if d < min_days and n < min_decisions:
            need.append(f"{min_days - d} jours ou {min_decisions - n} décisions d'observation")
        need.append("battre le champion et le 95e centile des modèles aléatoires en direct")
        return "Prétendant : observé en direct, ne parle pas encore. Pour être promu : " + " ; ".join(need) + "."
    if status == "retired":
        return "Retiré : remplacé par un prétendant qui a fait mieux. Ses livres sont conservés."
    return "Candidat : pas encore dans l'arène."


def money(delta: float, nav0: float = 10_000.0) -> str:
    """A P&L fraction as euros on the virtual book."""
    return f"{delta * nav0:+,.0f} €".replace(",", " ")


REGIME_FR = {"bull": "haussier", "bear": "baissier", "range": "sans tendance"}


def explain_robustness(rob: dict) -> list[str]:
    """The random-window robustness test (spec §8 step 7) in plain French."""
    if not rob:
        return []
    n = int(rob.get("n_windows", 0) or 0)
    lo = max(1, round(int(rob.get("min_days", 60)) / 30))
    hi = max(lo, round(int(rob.get("max_days", 180)) / 30))
    lines: list[str] = []
    wr = rob.get("win_rate_overall")
    head = f"Sur {n} périodes de {lo} à {hi} mois tirées au hasard"
    lines.append(head + (f" : gagne dans {float(wr) * 100:.0f} % des cas." if wr is not None else "."))
    rates = rob.get("win_rate_by_regime") or {}
    counts = rob.get("n_by_regime") or {}
    parts = []
    for reg, name in REGIME_FR.items():
        k = int(counts.get(reg, 0) or 0)
        r = rates.get(reg)
        if k == 0 or r is None:
            parts.append(f"{name} : aucune période")
        else:
            unit = "périodes" if reg == "bull" else ""
            parts.append(f"{name} {float(r) * 100:.0f} % ({k}{(' ' + unit) if unit else ''})")
    if parts:
        lines.append("Par type de marché : " + ", ".join(parts) + ".")
    med, worst = rob.get("median_sharpe"), rob.get("worst_return")
    if med is not None and worst is not None:
        lines.append(
            f"Rendement ajusté du risque médian {float(med):.2f} ; pire période : {float(worst) * 100:+.1f} %."
        )
    if "regimes_positive" in rob:
        pos, need = int(rob["regimes_positive"]), int(rob.get("n_regimes_required", 2))
        judged = int(rob.get("regimes_judged", len([c for c in counts.values() if c])))
        thr = float(rob.get("min_regime_win_rate", 0.5)) * 100
        mark = "" if rob.get("passed") is None else ("✓ " if rob["passed"] else "✗ ")
        lines.append(
            f"{mark}Tient dans {pos} types de marché sur {judged} (il en faut {need}, à au moins {thr:.0f} % de périodes gagnantes)."
        )
    return lines
