"""Dixon-Coles enrichi de la FORME (résidus orthogonaux), coefficients par MLE.

Étend le modèle de base avec deux termes additifs sur les log-taux :

    log(λ) = intercept + home_adv + att[h] − def[a] + γ·formeOff[h] + δ·formeDef[a]
    log(μ) = intercept           + att[a] − def[h] + γ·formeOff[a] + δ·formeDef[h]

  γ (gamma) : poids de la forme offensive (a-t-on marqué plus que prévu récemment ?)
  δ (delta) : poids de la forme défensive de l'adversaire.

γ et δ sont estimés par maximum de vraisemblance AVEC le reste. S'ils ressortent
≈ 0, la forme n'apporte rien — verdict honnête, pas de dégradation.

La forme par match (formeOff/formeDef de chaque équipe AVANT ce match) est fournie
en entrée : ce module ne la calcule pas (cf. signals/form.py), il l'intègre. Cela
garde la responsabilité de l'anti-fuite chez l'appelant (walk-forward).

Réutilise tau du modèle de base (une seule définition de la correction DC).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson

from model.dixoncoles import tau, MAX_GOALS


@dataclass
class FormParams:
    teams: list[str]
    attack: np.ndarray
    defence: np.ndarray
    home_adv: float
    rho: float
    intercept: float
    gamma: float          # poids forme offensive
    delta: float          # poids forme défensive

    def index(self, team: str) -> int:
        return self.teams.index(team)


def _unpack(theta, n):
    a_free = theta[:n - 1]
    d_free = theta[n - 1:2 * (n - 1)]
    attack = np.concatenate([a_free, [-a_free.sum()]])
    defence = np.concatenate([d_free, [-d_free.sum()]])
    home_adv = theta[2 * (n - 1)]
    rho = theta[2 * (n - 1) + 1]
    intercept = theta[2 * (n - 1) + 2]
    gamma = theta[2 * (n - 1) + 3]
    delta = theta[2 * (n - 1) + 4]
    return attack, defence, home_adv, rho, intercept, gamma, delta


def _neg_log_likelihood(theta, hi, ai, hg, ag, n,
                        foff_h, fdef_h, foff_a, fdef_a):
    attack, defence, home_adv, rho, intercept, gamma, delta = _unpack(theta, n)

    log_lam = (intercept + home_adv + attack[hi] - defence[ai]
               + gamma * foff_h + delta * fdef_a)
    log_mu = (intercept + attack[ai] - defence[hi]
              + gamma * foff_a + delta * fdef_h)
    lam = np.exp(log_lam)
    mu = np.exp(log_mu)

    ll = poisson.logpmf(hg, lam) + poisson.logpmf(ag, mu)
    t = tau(hg, ag, lam, mu, rho)
    if np.any(t <= 0):
        return 1e10
    ll = ll + np.log(t)
    return -np.sum(ll)


def fit_form(home_teams, away_teams, home_goals, away_goals,
             form_off_home, form_def_home, form_off_away, form_def_away):
    """Ajuste le modèle enrichi. Les `form_*` sont des tableaux alignés sur les
    matchs : forme de chaque équipe AVANT le match correspondant (fournie par
    l'appelant, sans fuite)."""
    teams = sorted(set(home_teams) | set(away_teams))
    n = len(teams)
    idx = {t: i for i, t in enumerate(teams)}
    hi = np.array([idx[t] for t in home_teams])
    ai = np.array([idx[t] for t in away_teams])
    hg = np.asarray(home_goals, dtype=int)
    ag = np.asarray(away_goals, dtype=int)
    foff_h = np.asarray(form_off_home, dtype=float)
    fdef_h = np.asarray(form_def_home, dtype=float)
    foff_a = np.asarray(form_off_away, dtype=float)
    fdef_a = np.asarray(form_def_away, dtype=float)

    theta0 = np.zeros(2 * (n - 1) + 5)
    theta0[2 * (n - 1)] = 0.25      # home_adv
    theta0[2 * (n - 1) + 1] = -0.1  # rho
    # gamma, delta initialisés à 0 : on part de "la forme ne compte pas".

    res = minimize(
        _neg_log_likelihood, theta0,
        args=(hi, ai, hg, ag, n, foff_h, fdef_h, foff_a, fdef_a),
        method="L-BFGS-B", options={"maxiter": 2000},
    )
    attack, defence, home_adv, rho, intercept, gamma, delta = _unpack(res.x, n)
    return FormParams(
        teams=teams, attack=attack, defence=defence,
        home_adv=float(home_adv), rho=float(rho), intercept=float(intercept),
        gamma=float(gamma), delta=float(delta),
    )


def predict_1x2_form(params: FormParams, home: str, away: str,
                     form_off_home=0.0, form_def_home=0.0,
                     form_off_away=0.0, form_def_away=0.0) -> dict:
    """Prédiction 1/N/2 avec les termes de forme du match à prédire."""
    i, j = params.index(home), params.index(away)
    lam = np.exp(params.intercept + params.home_adv
                 + params.attack[i] - params.defence[j]
                 + params.gamma * form_off_home + params.delta * form_def_away)
    mu = np.exp(params.intercept + params.attack[j] - params.defence[i]
                + params.gamma * form_off_away + params.delta * form_def_home)
    goals = np.arange(MAX_GOALS + 1)
    mat = np.outer(poisson.pmf(goals, lam), poisson.pmf(goals, mu))
    for x in (0, 1):
        for y in (0, 1):
            mat[x, y] *= tau(x, y, lam, mu, params.rho)
    mat /= mat.sum()
    return {
        "home": float(np.tril(mat, -1).sum()),
        "draw": float(np.trace(mat)),
        "away": float(np.triu(mat, 1).sum()),
               }
