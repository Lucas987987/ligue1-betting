"""Cœur du modèle : Dixon-Coles fréquentiste (baseline).

Définition UNIQUE du modèle (cf. ARCHITECTURE.md §5 : model/ scindé). Ce module
ne lit aucun fichier et n'écrit rien — il contient seulement les maths. Les
scripts fit/predict l'importent.

Modèle (cf. ARCHITECTURE.md §2) :
    log(λ) = intercept + home_adv + att[i] - def[j]      (buts domicile)
    log(μ) = intercept           + att[j] - def[i]      (buts extérieur)
    P(x,y) = τ(x,y) · Poisson(x|λ) · Poisson(y|μ)
avec la correction Dixon-Coles τ sur les 4 scores faibles (paramètre ρ).

Contrainte d'identifiabilité : somme(att)=0 et somme(def)=0, imposée en
paramétrant la dernière équipe comme -somme(des autres).

V1 : pas de pondération temporelle (tous les matchs à poids égal), pas de
bayésien. Ces extensions viendront après validation de cette baseline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson

MAX_GOALS = 10  # tronque la matrice des scores à 0..10 buts par équipe


# ---------------------------------------------------------------------- #
# Correction Dixon-Coles τ
# ---------------------------------------------------------------------- #
def tau(x, y, lam, mu, rho):
    """Facteur correctif sur les 4 scores faibles. Vectorisable."""
    x = np.asarray(x)
    y = np.asarray(y)
    out = np.ones(np.broadcast(x, y, lam, mu).shape, dtype=float)
    out = np.where((x == 0) & (y == 0), 1.0 - lam * mu * rho, out)
    out = np.where((x == 0) & (y == 1), 1.0 + lam * rho, out)
    out = np.where((x == 1) & (y == 0), 1.0 + mu * rho, out)
    out = np.where((x == 1) & (y == 1), 1.0 - rho, out)
    return out


# ---------------------------------------------------------------------- #
# Paramétrage : un vecteur plat <-> (attaque, défense, home_adv, rho)
# ---------------------------------------------------------------------- #
@dataclass
class DixonColesParams:
    teams: list[str]
    attack: np.ndarray          # taille n_teams, somme = 0
    defence: np.ndarray         # taille n_teams, somme = 0
    home_adv: float
    rho: float
    intercept: float

    def index(self, team: str) -> int:
        return self.teams.index(team)


def _pack(attack_free, defence_free, home_adv, rho, intercept):
    """Vecteur d'optimisation : on optimise n-1 attaques et n-1 défenses
    (la dernière est déterminée par la contrainte somme=0)."""
    return np.concatenate(
        [attack_free, defence_free, [home_adv, rho, intercept]]
    )


def _unpack(theta, n_teams):
    a_free = theta[: n_teams - 1]
    d_free = theta[n_teams - 1 : 2 * (n_teams - 1)]
    home_adv, rho, intercept = theta[-3], theta[-2], theta[-1]
    # Dernière équipe = -somme des autres (contrainte d'identifiabilité).
    attack = np.concatenate([a_free, [-a_free.sum()]])
    defence = np.concatenate([d_free, [-d_free.sum()]])
    return attack, defence, home_adv, rho, intercept


# ---------------------------------------------------------------------- #
# Log-vraisemblance négative (à minimiser)
# ---------------------------------------------------------------------- #
def _neg_log_likelihood(theta, home_idx, away_idx, hg, ag, n_teams, weights):
    attack, defence, home_adv, rho, intercept = _unpack(theta, n_teams)

    log_lam = intercept + home_adv + attack[home_idx] - defence[away_idx]
    log_mu = intercept + attack[away_idx] - defence[home_idx]
    lam = np.exp(log_lam)
    mu = np.exp(log_mu)

    # Log-vraisemblance Poisson pour chaque match (buts observés hg, ag).
    ll = poisson.logpmf(hg, lam) + poisson.logpmf(ag, mu)

    # Correction Dixon-Coles (seulement sur les 4 scores faibles).
    t = tau(hg, ag, lam, mu, rho)
    # Garde-fou : τ doit rester > 0 (sinon log explose -> on pénalise fort).
    if np.any(t <= 0):
        return 1e10
    ll = ll + np.log(t)

    # Pondération temporelle Dixon-Coles : chaque match pèse weights[i] = φ(t).
    # weights tous à 1 => modèle non pondéré (rétrocompatible).
    return -np.sum(weights * ll)


# ---------------------------------------------------------------------- #
# Estimation
# ---------------------------------------------------------------------- #
def fit(home_teams, away_teams, home_goals, away_goals, weights=None) -> DixonColesParams:
    """Estime les paramètres par maximum de vraisemblance.

    Args:
        home_teams, away_teams : listes de noms canoniques.
        home_goals, away_goals : buts (entiers).
        weights : poids par match (ex. décroissance temporelle φ(t)=exp(-ξ·Δt)).
                  Si None, tous les matchs pèsent 1 (modèle non pondéré).
    """
    teams = sorted(set(home_teams) | set(away_teams))
    n = len(teams)
    idx = {t: i for i, t in enumerate(teams)}

    home_idx = np.array([idx[t] for t in home_teams])
    away_idx = np.array([idx[t] for t in away_teams])
    hg = np.asarray(home_goals, dtype=int)
    ag = np.asarray(away_goals, dtype=int)
    if weights is None:
        weights = np.ones(len(hg))
    else:
        weights = np.asarray(weights, dtype=float)

    # Initialisation neutre.
    theta0 = _pack(
        np.zeros(n - 1), np.zeros(n - 1),
        home_adv=0.25, rho=-0.1, intercept=0.0,
    )

    res = minimize(
        _neg_log_likelihood,
        theta0,
        args=(home_idx, away_idx, hg, ag, n, weights),
        method="L-BFGS-B",
        options={"maxiter": 1000},
    )

    attack, defence, home_adv, rho, intercept = _unpack(res.x, n)
    return DixonColesParams(
        teams=teams, attack=attack, defence=defence,
        home_adv=float(home_adv), rho=float(rho), intercept=float(intercept),
    )


# ---------------------------------------------------------------------- #
# Prédiction : matrice des scores -> P(1/N/2)
# ---------------------------------------------------------------------- #
def score_matrix(params: DixonColesParams, home: str, away: str) -> np.ndarray:
    """Matrice (MAX_GOALS+1)² des probabilités de chaque score exact."""
    i, j = params.index(home), params.index(away)
    lam = np.exp(params.intercept + params.home_adv + params.attack[i] - params.defence[j])
    mu = np.exp(params.intercept + params.attack[j] - params.defence[i])

    goals = np.arange(MAX_GOALS + 1)
    px = poisson.pmf(goals, lam)          # buts domicile
    py = poisson.pmf(goals, mu)           # buts extérieur
    mat = np.outer(px, py)                # indépendance Poisson

    # Applique τ sur les 4 coins bas-gauche.
    for x in (0, 1):
        for y in (0, 1):
            mat[x, y] *= tau(x, y, lam, mu, params.rho)

    # Renormalisation (τ perturbe légèrement la somme).
    return mat / mat.sum()


def predict_1x2(params: DixonColesParams, home: str, away: str) -> dict[str, float]:
    """Probabilités des 3 issues. somme = 1."""
    mat = score_matrix(params, home, away)
    p_home = float(np.tril(mat, -1).sum())   # buts_dom > buts_ext
    p_draw = float(np.trace(mat))            # diagonale
    p_away = float(np.triu(mat, 1).sum())    # buts_dom < buts_ext
    return {"home": p_home, "draw": p_draw, "away": p_away}
