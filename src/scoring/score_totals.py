"""Scoring Over/Under — EV + Signal × Fiabilité par seuil coté.

Réutilise les briques de value.py (devig, composantes de fiabilité, plancher).
Un marché over/under est un marché à 2 issues (over, under) ; la mécanique est
celle du 1/N/2, juste sur 2 issues. La Fiabilité combine :
  - certitude bayésienne (largeur de l'intervalle de crédibilité de la proba)
  - convergence des books (dispersion des cotes over/under)
  - qualité des données (nombre de matchs des deux équipes)

On ne score QUE les seuils réellement cotés (EV/Signal impossibles sans cote).
"""

from __future__ import annotations

from dataclasses import dataclass

from scoring.value import (
    devig, reliability_from_ci, reliability_from_books, reliability_from_data,
    RELIABILITY_FLOOR,
)


@dataclass
class OUIssueScore:
    threshold: float
    side: str            # 'over' | 'under'
    p_model: float
    p_market: float
    odds: float
    ev: float
    signal: float
    reliability: float
    score: float


@dataclass
class OUMatchScore:
    home: str
    away: str
    issues: list[OUIssueScore]   # over + under pour chaque seuil coté

    def best_by_score(self):
        return max(self.issues, key=lambda s: s.score) if self.issues else None

    def best_by_ev(self):
        return max(self.issues, key=lambda s: s.ev) if self.issues else None


def score_over_under(
    home: str,
    away: str,
    ou_pred: dict,                 # sortie de predict_over_under : {seuil: {over, under, over_ci, ...}}
    ou_odds: dict,                 # sortie de extract_totals : {seuil: {over, under, over_prices, under_prices}}
    n_matches: dict[str, int] | None = None,
) -> OUMatchScore:
    """Score chaque seuil présent À LA FOIS dans le modèle et dans les cotes."""
    n_home = (n_matches or {}).get(home, 60)
    n_away = (n_matches or {}).get(away, 60)
    rel_data = reliability_from_data(n_home, n_away)
    rel_data_f = RELIABILITY_FLOOR + (1 - RELIABILITY_FLOOR) * rel_data

    issues: list[OUIssueScore] = []
    # Seuils cotés ET prédits.
    for seuil in sorted(set(ou_pred) & set(ou_odds)):
        pred = ou_pred[seuil]
        odds = ou_odds[seuil]
        # Probas marché dévigorisées (2 issues : over/under).
        p_market = devig({"over": odds["over"], "under": odds["under"]})

        for side in ("over", "under"):
            p_m = pred[side]
            p_k = p_market[side]
            cote = odds[side]
            ev = p_m * cote - 1.0
            signal = (p_m - p_k) / p_k if p_k > 0 else 0.0

            # Fiabilité : certitude bayésienne (intervalle) × convergence × données.
            ci = pred.get(f"{side}_ci")
            rel_ci = reliability_from_ci(ci["lo"], ci["hi"]) if ci else 0.5
            rel_ci = RELIABILITY_FLOOR + (1 - RELIABILITY_FLOOR) * rel_ci
            prices = odds.get(f"{side}_prices", [])
            rel_books = reliability_from_books(prices)
            rel_books = RELIABILITY_FLOOR + (1 - RELIABILITY_FLOOR) * rel_books
            reliability = rel_ci * rel_books * rel_data_f

            score = max(signal, 0.0) * reliability
            issues.append(OUIssueScore(
                threshold=seuil, side=side,
                p_model=p_m, p_market=p_k, odds=cote,
                ev=ev, signal=signal, reliability=reliability, score=score,
            ))

    return OUMatchScore(home=home, away=away, issues=issues)
