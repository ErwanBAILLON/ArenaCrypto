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
    "funding_skew": FamilyCard(
        "funding_skew",
        "Foule à levier (écart de funding)",
        "Vend les perpétuels dont le funding est très au-dessus de celui des autres, achète ceux dont il est très en dessous, "
        "à parts égales des deux côtés. Le funding est le prix du levier : celui qui paie beaucoup plus que les autres est le "
        "côté encombré du marché, donc celui qu'une cascade de liquidations nettoie en premier.",
        [
            "Funding Binance (8 h) sur 7 jours, en écart-type par rapport au reste de l'univers",
            "Cours 1 h pour valoriser",
        ],
        "Une fois par semaine, le lundi à 00:00 UTC",
        [
            "Ne couvre rien : contrairement au carry, la position est directionnelle.",
            "Encombrement et tendance vont souvent ensemble : une partie du pari est un simple retour à la moyenne, "
            "et le test d'entrée ne sait pas les distinguer.",
        ],
    ),
    "crowded_trend": FamilyCard(
        "crowded_trend",
        "Tendance non encombrée",
        "Même signal de tendance que la famille « Tendance », mais on jette toute position que la foule a déjà prise : pas "
        "d'achat sur un actif dont les acheteurs à levier paient déjà une prime, pas de vente sur un actif dont les vendeurs "
        "sont déjà payés. Une tendance encombrée se termine en liquidations, pas en essoufflement.",
        [
            "Cours 1 h : EMA 50 et 200, rendements 30 et 90 jours, volatilité 30 jours, ATR 14",
            "Funding Binance (8 h) sur 7 jours, en écart-type par rapport au reste de l'univers",
        ],
        "Une fois par jour à 00:00 UTC, tient entre-temps",
        [
            "Si le funding ne filtre rien d'utile, cette famille est « Tendance » en plus cher en frais : c'est précisément "
            "ce que la comparaison des trois familles doit trancher.",
        ],
    ),
    "xs_sparse": FamilyCard(
        "xs_sparse",
        "Panier simple (coupe transversale)",
        "Classe tous les actifs de l'univers sur cinq critères simples (momentum, retour à la moyenne à une semaine, "
        "encombrement du funding, volatilité, position dans le range), fait la moyenne des classements et achète le haut "
        "en vendant le bas, à parts égales. Rien n'est estimé sur les données : c'est le témoin qui sert à savoir si un "
        "modèle compliqué apporte vraiment quelque chose.",
        [
            "Le panel complet de la coupe transversale (rangs uniquement)",
            "Momentum 30 jours en sautant la dernière semaine, rendement 7 jours, funding, volatilité 30 jours",
        ],
        "Une fois par semaine le lundi, mais chaque position peut se fermer tous les jours sur son objectif de gain",
        [
            "Neutre au marché par construction : ne peut pas gagner en étant simplement long crypto.",
            "Aucun paramètre appris, donc très peu d'essais consommés : c'est son avantage face au test d'entrée.",
        ],
    ),
    "xs_complex": FamilyCard(
        "xs_complex",
        "Panier appris (milliers de paramètres)",
        "Même univers, mêmes règles de sortie et même construction de portefeuille que le panier simple, mais la note de "
        "chaque actif vient d'un modèle à plusieurs milliers de paramètres entraîné sur l'historique. Ce qu'on lui demande "
        "d'apprendre, ce sont les interactions qu'une moyenne ne peut pas dire : la réversion paie quand le funding est "
        "extrême, le momentum paie quand les actifs se dispersent.",
        [
            "Le panel complet de la coupe transversale (une centaine de colonnes brutes et leurs rangs)",
            "Un modèle entraîné hors ligne, rejoué depuis un artefact stocké",
        ],
        "Une fois par semaine le lundi, avec fermeture possible chaque jour sur l'objectif de gain",
        [
            "Sans artefact entraîné, il reste à plat : pas de repli silencieux sur autre chose.",
            "La thèse qu'il teste est contestée (Nagel 2025) ; le panier simple existe précisément pour l'arbitrer.",
        ],
    ),
    "null_neutral": FamilyCard(
        "null_neutral",
        "Repère : aléatoire neutre",
        "Tire au hasard autant d'achats que de ventes, aux mêmes tailles et au même rythme que les paniers en coupe "
        "transversale. Le repère aléatoire classique n'est pas neutre et perd surtout des frais ; celui-ci isole la "
        "seule chose qu'un panier prétend faire : choisir.",
        ["Rien : c'est le hasard"],
        "Nouveau tirage chaque semaine",
    ),
    "xs_adaptive": FamilyCard(
        "xs_adaptive",
        "Panier adaptatif (signaux gagnés)",
        "Garde une longue liste d'indicateurs candidats et, chaque semaine, ne fait voter que ceux dont la corrélation "
        "avec les rendements relatifs des 52 semaines précédentes est significative, pondérés par cette corrélation. "
        "Les signaux changent de signe avec le régime ; ce panier change de signaux avec eux, sans jamais regarder la "
        "période qu'il trade.",
        [
            "Le panel complet : volatilité, effet loterie (gains extrêmes, asymétrie), tendance, funding, liquidité",
            "Ses propres panels passés, pour mesurer ce qui a marché récemment",
        ],
        "Une fois par semaine le lundi, fermeture possible chaque jour sur l'objectif de gain",
        ["S'il n'y a aucun signal significatif, il reste à plat plutôt que de deviner."],
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
    "price_action": FamilyCard(
        "price_action",
        "Analyse technique classique (price action)",
        "Trace mécaniquement les supports et résistances (plus hauts et plus bas de swing), achète une cassure de résistance confirmée par le volume "
        "et un chandelier haussier, ou un repli de Fibonacci sur la dernière impulsion marqué par un chandelier de retournement (marteau, englobante) ; symétrique à la vente. "
        "Horizon swing : quelques jours à quelques semaines ; horizon position : plusieurs mois.",
        [
            "Cours natifs (1 h crypto, 1 j classique) : plus hauts et plus bas de swing sur la période de repérage",
            "Volume comparé à la médiane de la période",
            "Forme du dernier chandelier (ouverture, plus haut, plus bas, clôture)",
            "ATR 14 pour graduer la force de la cassure",
        ],
        "Une fois par jour à 00:00 UTC (swing) ou le lundi à 00:00 UTC (position), tient jusqu'au stop",
        [
            "Les niveaux d'un chartiste sont subjectifs ; on les rend mécaniques (extrema locaux), ce qui n'est qu'une lecture parmi d'autres.",
            "Le backtest ne voit pas les mèches intra-barre : un stop touché puis repris dans la même barre n'est pas détecté.",
        ],
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
    "bench_hold": FamilyCard(
        "bench_hold",
        "Repère : garder l'actif de référence",
        "Achète l'actif de référence de l'univers (par exemple le S&P 500) et ne bouge plus. La question de base : le modèle fait-il mieux que simplement le détenir ?",
        ["Cours de l'actif de référence"],
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
    elif family == "price_action":
        style = str(p.get("style", "swing"))
        horizon = (
            "position (plusieurs mois, re-décision le lundi)"
            if style == "position"
            else "swing (jours à semaines, re-décision quotidienne)"
        )
        out += [
            f"horizon {horizon}",
            f"pivots sur ±{p.get('pivot_days', 10)} jours, niveaux repérés sur {p.get('level_lookback_days', 60)} jours",
            f"cassure au-delà de {float(p.get('breakout_buffer', 0.002)) * 100:.1f} % du niveau avec volume > {float(p.get('min_volume_ratio', 1.2)):.1f}× la médiane",
            f"repli de Fibonacci entre {float(p.get('fib_low', 0.382)) * 100:.1f} % et {float(p.get('fib_high', 0.618)) * 100:.1f} %",
            f"stop au plus bas / plus haut des {p.get('stop_days', 20)} derniers jours",
            f"jusqu'à {p.get('k', 4)} positions de {float(p.get('max_weight', 0.25)) * 100:.0f} % chacune",
        ]
    elif family == "news":
        out += [
            f"sentiment 7 jours > {p.get('min_sent_7d', 0.1)}",
            f"au moins {p.get('min_n_7d', 3)} articles sur 7 jours",
            f"jusqu'à {p.get('k', 3)} actifs",
        ]
    return out
