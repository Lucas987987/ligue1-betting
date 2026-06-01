"""Forme récente mesurée comme RÉSIDU vs le modèle (orthogonale aux forces).

Le piège de la forme : "gagner ses 5 derniers matchs" reflète surtout la QUALITÉ
de l'équipe, que les forces du modèle capturent déjà. Recompter ça doublonnerait.

La SEULE information nouvelle est la SUR/SOUS-PERFORMANCE récente par rapport à ce
que le modèle attendait. Une équipe "en forme" = qui marque PLUS / encaisse MOINS
que ce que sa force prédisait. Une équipe qui gagne exactement comme prévu n'est
pas "en forme" au sens informatif : elle est juste bonne, et le modèle le sait.

Définition :
  Pour un match passé d'une équipe :
    - buts attendus en attaque  = λ (selon les forces + avantage terrain)
    - buts attendus en défense  = μ (ce qu'elle devait encaisser)
    surperf_off = buts_marqués_réels   − λ
    surperf_def = μ − buts_encaissés_réels      (positif = a mieux défendu que prévu)
  Forme d'une équipe avant un match = moyenne de ces écarts sur ses N derniers
  matchs joués (offensive et défensive séparées).

ANTI-FUITE (critique) : les buts attendus d'un match doivent être calculés avec un
modèle entraîné UNIQUEMENT sur le passé strict de ce match. Ce module ne fait PAS
le fit lui-même : il reçoit une fonction `expected_goals(date, home, away)` que
l'appelant garantit sans fuite (cf. walk-forward). Ainsi la responsabilité de
l'absence de fuite est explicite et testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np


@dataclass
class MatchOutcome:
    """Un match joué, du point de vue du calcul de forme."""
    date: datetime
    home: str
    away: str
    home_goals: int
    away_goals: int


@dataclass
class TeamForm:
    """Forme d'une équipe à un instant donné (moyenne des résidus récents)."""
    off: float    # surperformance offensive moyenne (>0 = marque plus que prévu)
    deff: float   # surperformance défensive moyenne (>0 = encaisse moins que prévu)
    n: int        # nombre de matchs réellement utilisés (peut être < window)


def expected_goals_from_params(params, home: str, away: str) -> tuple[float, float]:
    """λ, μ pour un match, depuis un modèle Dixon-Coles déjà ajusté.

    Réplique la formule de dixoncoles.predict (une seule définition logique).
    λ = buts attendus de l'équipe à domicile ; μ = ceux de l'équipe extérieure.
    """
    i = params.teams.index(home)
    j = params.teams.index(away)
    lam = np.exp(params.intercept + params.home_adv
                 + params.attack[i] - params.defence[j])
    mu = np.exp(params.intercept + params.attack[j] - params.defence[i])
    return float(lam), float(mu)


def compute_form(
    target_date: datetime,
    team: str,
    history: list[MatchOutcome],
    expected_goals,            # callable(date, home, away) -> (lam, mu) SANS FUITE
    window: int = 5,
) -> TeamForm:
    """Forme d'une `team` juste avant `target_date`.

    On prend ses `window` derniers matchs joués STRICTEMENT avant target_date,
    et on moyenne les résidus (réel − attendu). `expected_goals` doit être fourni
    par l'appelant et ne doit utiliser que l'information disponible à la date du
    match considéré (responsabilité d'anti-fuite déléguée et testée séparément).
    """
    # Matchs passés de l'équipe, du plus récent au plus ancien.
    past = [m for m in history
            if m.date < target_date and (m.home == team or m.away == team)]
    past.sort(key=lambda m: m.date, reverse=True)
    recent = past[:window]

    if not recent:
        return TeamForm(off=0.0, deff=0.0, n=0)  # pas d'historique => forme neutre

    off_res, def_res = [], []
    for m in recent:
        lam, mu = expected_goals(m.date, m.home, m.away)
        if team == m.home:
            scored, conceded = m.home_goals, m.away_goals
            exp_scored, exp_conceded = lam, mu
        else:
            scored, conceded = m.away_goals, m.home_goals
            exp_scored, exp_conceded = mu, lam
        off_res.append(scored - exp_scored)          # >0 : a marqué plus que prévu
        def_res.append(exp_conceded - conceded)      # >0 : a encaissé moins que prévu

    return TeamForm(
        off=float(np.mean(off_res)),
        deff=float(np.mean(def_res)),
        n=len(recent),
    )
