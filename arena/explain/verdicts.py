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
