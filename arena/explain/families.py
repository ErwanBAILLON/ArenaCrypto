"""What each family does and what it looks at, in plain French."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class FamilyCard:
    key: str
    title: str
    summary: str  # two sentences: the idea and the economic reason
    inputs: list[str]  # data the model reads
    cadence: str  # when it re-decides
    caveats: list[str] = field(default_factory=list)


CARDS: dict[str, FamilyCard] = {
    "carry": FamilyCard(
        "carry",
        "Carry de funding",
        "Encaisse le funding des contrats perpétuels : quand les acheteurs à levier paient les vendeurs toutes les 8 heures, "
        "le modèle prend la position vendeuse sur le perpétuel et la couvre au comptant. Le gain vient du funding, pas du prix.",
        [
            "Taux de funding Binance (8 h) sur 14 jours",
            "Funding horaire Hyperliquid (instantané)",
            "Cours 1 h pour valoriser",
        ],
        "Une fois par semaine, le lundi à 00:00 UTC",
        ["Modèle de coût idéalisé : couverture parfaite, sans risque de basis ni coût d'emprunt du comptant."],
    ),
    "trend_ts": FamilyCard(
        "trend_ts",
        "Tendance (momentum temporel)",
        "Suit la tendance de chaque actif : achète ce qui monte depuis 1 à 3 mois, vend ce qui baisse, et dimensionne chaque "
        "position pour viser une volatilité cible. Un actif qui monte depuis des mois a tendance à continuer (Moskowitz, Ooi, Pedersen 2012).",
        [
            "Cours 1 h : moyennes mobiles exponentielles 50 et 200",
            "Rendements sur 30 et 90 jours",
            "Volatilité réalisée 30 jours",
            "ATR 14 pour le stop",
        ],
        "Une fois par jour à 00:00 UTC, tient entre-temps",
        ["Perd dans les marchés sans tendance : les entrées et sorties se paient en frais."],
    ),
    "xs_momentum": FamilyCard(
        "xs_momentum",
        "Momentum relatif (classement)",
        "Classe les actifs par performance sur 30 jours (en ignorant les 2 derniers), achète les meilleurs et vend les pires. "
        "La prime de momentum relatif est documentée sur les actions depuis les années 90 et se retrouve en crypto.",
        ["Cours 1 h de tous les actifs de l'univers", "Rendement 30 jours décalé de 2 jours"],
        "Re-classement hebdomadaire avec tolérance de rang, pour ne pas churner",
    ),
    "regime": FamilyCard(
        "regime",
        "Régime de marché",
        "Classe le marché selon la tendance du BTC et sa volatilité (haussier calme, haussier volatil, baissier, sans tendance) "
        "et applique une allocation fixe par état : tendance en marché haussier, carry en marché sans tendance, cash en marché baissier volatil.",
        [
            "Cours 1 h du BTC : EMA 50 et 200",
            "Volatilité réalisée 30 jours et ses terciles sur 180 jours",
            "Funding pour la poche carry",
        ],
        "Une fois par jour à 00:00 UTC",
        ["Version à règles fixes ; une version apprise (classifieur) est prévue."],
    ),
    "meta_label": FamilyCard(
        "meta_label",
        "Méta-étiquetage (filtre appris)",
        "Ne choisit pas les positions : il reprend les signaux de la tendance et du momentum relatif et un modèle LightGBM décide "
        "s'il faut les prendre et à quelle taille, selon le contexte (volatilité, régime, funding, intérêt ouvert, actualité, heure).",
        [
            "Signaux des familles tendance et momentum",
            "Volatilité 30 j, régime, funding 3 j, variation d'intérêt ouvert 24 h",
            "Sentiment actualité 24 h et 7 j",
            "Calendrier macro",
        ],
        "À chaque heure sur les signaux de base ; modèle ré-entraîné les 1er et 15 du mois",
        ["Sans modèle admis, il se contente de moyenner ses bases (mode pass-through)."],
    ),
    "news": FamilyCard(
        "news",
        "Actualité (sentiment lent)",
        "Lit les flux RSS crypto, note chaque article (ton, type d'événement, actifs concernés) et achète les actifs dont le sentiment "
        "sur 7 jours est positif et en hausse. Ce qui survit aux gros titres, c'est la dérive lente d'un récit sur plusieurs jours.",
        [
            "6 flux RSS (CoinDesk, CoinTelegraph, The Block, Decrypt, Bitcoin Magazine, The Defiant)",
            "Scores VADER + lexique crypto, agrégés 24 h et 7 j",
        ],
        "À chaque heure, positions conservées tant que le sentiment tient",
        ["Jamais backtesté : ses scores n'existent qu'en direct, il est jugé uniquement dans l'arène."],
    ),
    "null_cash": FamilyCard(
        "null_cash",
        "Repère : ne rien faire",
        "Reste en cash. Le plancher : tout modèle utile doit faire mieux.",
        ["Aucune"],
        "Jamais",
    ),
    "null_random": FamilyCard(
        "null_random",
        "Repère : aléatoire",
        "Tire des positions au hasard une fois par semaine. Cinq copies avec des graines différentes donnent la distribution de ce qu'obtient la chance ; "
        "un modèle qui ne dépasse pas leur 95e centile n'a rien trouvé.",
        ["Aucune (générateur aléatoire déterministe)"],
        "Redistribution hebdomadaire",
    ),
    "bench_btc_hold": FamilyCard(
        "bench_btc_hold",
        "Repère : garder du BTC",
        "Achète du BTC et ne bouge plus. C'est la question de base : le modèle fait-il mieux que simplement tenir du bitcoin ?",
        ["Cours du BTC"],
        "Jamais",
    ),
    "bench_carry_equal": FamilyCard(
        "bench_carry_equal",
        "Repère : carry sur tout",
        "Prend le carry de funding sur tous les actifs à poids égal, sans sélection. Mesure ce que la sélection du modèle carry ajoute.",
        ["Funding de tous les actifs"],
        "Jamais",
    ),
}

STATUS_FR = {"champion": "champion", "challenger": "prétendant", "candidate": "candidat", "retired": "retiré"}
ROLE_FR = {"null": "repère aléatoire", "benchmark": "repère", "competitor": "modèle"}


def card(family: str) -> FamilyCard:
    return CARDS.get(family, FamilyCard(family, family, "Famille non documentée.", [], "inconnue"))


def params_in_words(family: str, params: dict[str, Any]) -> list[str]:
    """Readable rendering of the parameters that matter for humans."""
    p = params
    out: list[str] = []
    if family == "carry":
        out += [
            f"funding moyen sur {p.get('lookback_days', 14)} jours",
            f"jusqu'à {p.get('k', 5)} actifs",
            f"seuil d'entrée {float(p.get('min_rate', 0.00005)) * 100:.4f} % par 8 h (≈ {float(p.get('min_rate', 0.00005)) * 3 * 365 * 100:.1f} %/an)",
            f"sortie sous {float(p.get('exit_ratio', 0.3)) * 100:.0f} % du seuil",
        ]
    elif family == "trend_ts":
        out += [
            f"EMA {p.get('fast', 50)} / {p.get('slow', 200)}",
            f"rendements {p.get('lb_short_days', 30)} et {p.get('lb_long_days', 90)} jours",
            f"volatilité cible {float(p.get('target_vol', 0.2)) * 100:.0f} %/an par position",
            f"stop à {p.get('atr_stop_mult', 3.0)} ATR",
        ]
    elif family == "xs_momentum":
        out += [
            f"classement sur {p.get('lookback_days', 30)} jours, {p.get('skip_days', 2)} jours ignorés",
            f"{p.get('k', 3)} achetés / {p.get('k', 3)} vendus",
            "achats seulement" if p.get("long_only") else "achats et ventes",
            f"tolérance de rang {p.get('band', 1)}",
        ]
    elif family == "regime":
        out += [
            f"tendance EMA {p.get('trend_fast', 50)} / {p.get('trend_slow', 200)} du BTC",
            f"volatilité sur {int(p.get('vol_window_days', 30))} jours",
            f"{float(p.get('bull_weight', 0.5)) * 100:.0f} % investi en marché haussier",
            f"{float(p.get('range_carry_weight', 0.5)) * 100:.0f} % en carry en marché sans tendance",
        ]
    elif family == "meta_label":
        bases = ", ".join(b.get("family", "?") for b in p.get("bases", []))
        out += [
            f"bases : {bases}",
            f"seuil de confiance {float(p.get('threshold', 0.55)) * 100:.0f} %",
            "modèle appris chargé" if p.get("model_str") else "sans modèle (pass-through)",
        ]
    elif family == "news":
        out += [
            f"sentiment 7 jours > {p.get('min_sent_7d', 0.1)}",
            f"au moins {p.get('min_n_7d', 3)} articles sur 7 jours",
            f"jusqu'à {p.get('k', 3)} actifs",
        ]
    return out
