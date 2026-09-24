# Sorties en direct : stops et objectifs exécutés entre deux ticks

*2026-09-24*

## Ce qui change

L'arène décidait tout à l'heure pleine, sur bougies closes. Un stop touché à
14:20 et revenu à 14:59 n'existait pas ; un stop touché à 14:20 qui continuait
de tomber était payé à la clôture de 15:00. Un processus continu, `arena live`
(Deployment `arena-live`, 1 réplique), lit désormais les prix perp Binance
toutes les 5 s et exécute les règles de **sortie** au moment où elles sont
franchies. Rien n'ouvre de position ni ne re-sélectionne un livre en dehors du
tick : le protocole de décision reste horaire, seul le respect des règles de
sortie devient continu.

## Règles honorées

| Famille | Règle | Mesure |
|---|---|---|
| `LadderHoldingCompetitor` (`xs_sparse`, `xs_complex`, `xs_adaptive`) | leur stop (8 %) et leur échelle de ROI, tels que paramétrés | excès de rendement sur le panier figé à l'entrée, comme au tick, recalculé sur prix live pour la jambe et pour le panier |
| tout compétiteur avec `live_stop` et/ou `live_roi` dans ses params | stop en fraction, échelle `[[barres, cible], …]` | rendement brut de la jambe depuis son ouverture |

Le barreau « deadline » d'une échelle (tenir au plus N barres) reste au tick :
c'est une fonction du temps, pas du prix.

Donner `live_stop`/`live_roi` à une famille existante = changement de
paramètres = **nouvelle version de challenger**, jamais une édition du
champion en place.

## Mécanique d'une sortie

1. `fills` : une ligne au prix streamé (`weight_before`, `price`, `reason`, `excess`).
2. `targets` : le livre entier est **ré-énoncé** à l'instant du fill, jambes
   survivantes recopiées, jambe close à 0 (`reason.live_exit`). `last_targets`
   lit « toutes les lignes au dernier ts » : une ligne isolée aurait effacé le
   reste du livre.
3. `competitor_state` : `live_close(symbol, reason)` retire la jambe du livre
   tenu et pose un cooldown (`live_cooldown_bars`, 24 par défaut ; 24 barres
   chez les familles ladder) pour que le rebalancement suivant ne rachète pas
   aussitôt ce qu'on vient de couper.
4. Alerte `live_exit` (Telegram, hors quota journalier).
5. Au tick suivant, `Book.step(..., fills=…)` gagne `w × (fill / clôture_préc − 1)`
   et paie le coût de sortie (frais + impact selon la liquidité du symbole),
   puis marque le fill `booked_ts`. Le funding de l'heure partielle est ignoré.

## Garde-fous

- **Fenêtre de silence** autour de chaque minute de tick (`--tick-minutes 5,35`,
  de −1 à +5 min) : un fill écrit pendant le tick serait horodaté après la
  décision du tick et lu comme le livre courant.
- Un fill par compétiteur et par passe ; le livre est relu avant la suivante.
- Prix manquant = pas de signal. Univers non Binance (ETF Yahoo) = non surveillé.
- Erreur sur un compétiteur isolée (rollback), comme au tick.

## Asymétrie assumée backtest ↔ direct

Le juge (`arena/judge/backtest.py`) évalue toujours les stops sur clôtures
horaires. En direct, la sortie est plus tôt et à un prix différent (souvent
moins mauvais pour un stop, plus tôt pour un objectif). Le backtest est donc
légèrement **pessimiste** sur les stops et **conservateur** sur les objectifs
touchés en intra-heure. Aligner le backtest supposerait de modéliser le
franchissement intra-barre (high/low), ce qui reste une approximation : la
mesure des familles ladder est un excès sur panier, non observable en OHLC.
Choix : garder le backtest tel quel, comparer les fills réels à ce que le tick
aurait fait, et décider sur données.

## Tableau de bord

`/api/live` change de `version` dès qu'un fill est écrit (poll 10 s) ; le fil
« Achats et ventes » affiche « sortie · stop en direct à <prix> » ou
« objectif atteint en direct ». La valeur live du tableau des agents reflète
immédiatement le livre ré-énoncé.
