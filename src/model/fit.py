"""Étage 3a (V1 fréquentiste) — Ajuste le Dixon-Coles sur matches.csv.

Lit data/processed/matches.csv (sortie de l'étage 2), ajuste le modèle sur tous
les matchs JOUÉS (ceux qui ont un score), et écrit :
  - data/model/params.json      : paramètres estimés (forces, home_adv, rho)
  - data/model/team_ratings.csv : classement lisible des forces par équipe
  - data/model/sample_predictions.csv : prédictions 1/N/2 pour quelques affiches

V1 : modèle fréquentiste, tous matchs à poids égal. Pas de découplage fit/predict
sophistiqué encore (le bayésien l'introduira). Ici fit + échantillon de prédictions
dans le même script, pour avoir un premier résultat tangible.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from model.dixoncoles import DixonColesParams, fit, predict_1x2

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MATCHES = _ROOT / "data" / "processed" / "matches.csv"
DEFAULT_MODEL_DIR = _ROOT / "data" / "model"

HOME = "HomeTeamCanonical"
AWAY = "AwayTeamCanonical"
HG = "FTHG"   # Full Time Home Goals
AG = "FTAG"   # Full Time Away Goals


def load_played_matches(matches_path: Path):
    """Charge les matchs joués (score présent, équipes résolues)."""
    home, away, hg, ag = [], [], [], []
    skipped = 0
    with matches_path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            h, a = (row.get(HOME) or "").strip(), (row.get(AWAY) or "").strip()
            gh, ga = (row.get(HG) or "").strip(), (row.get(AG) or "").strip()
            # On ignore : équipe non résolue, ou match non joué (score vide).
            if not h or not a or not gh or not ga:
                skipped += 1
                continue
            try:
                home.append(h); away.append(a)
                hg.append(int(float(gh))); ag.append(int(float(ga)))
            except ValueError:
                skipped += 1
    return home, away, hg, ag, skipped


def params_to_dict(p: DixonColesParams) -> dict:
    return {
        "teams": p.teams,
        "attack": p.attack.tolist(),
        "defence": p.defence.tolist(),
        "home_adv": p.home_adv,
        "rho": p.rho,
        "intercept": p.intercept,
    }


def run(matches_path: Path = DEFAULT_MATCHES, model_dir: Path = DEFAULT_MODEL_DIR):
    home, away, hg, ag, skipped = load_played_matches(matches_path)
    n = len(hg)
    if n < 100:
        raise ValueError(
            f"Trop peu de matchs joués ({n}) pour un fit fiable. "
            f"Vérifier matches.csv et la résolution des noms."
        )

    print(f"Matchs joués utilisés : {n}  (ignorés : {skipped})")
    params = fit(home, away, hg, ag)
    print(f"Équipes : {len(params.teams)}")
    print(f"home_adv = {params.home_adv:.3f} | rho = {params.rho:.3f} | "
          f"intercept = {params.intercept:.3f}")

    model_dir.mkdir(parents=True, exist_ok=True)

    # 1. Paramètres bruts.
    (model_dir / "params.json").write_text(
        json.dumps(params_to_dict(params), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # 2. Classement lisible des forces (attaque - défense = force nette).
    ratings = sorted(
        zip(params.teams, params.attack, params.defence),
        key=lambda t: -(t[1] + t[2]),  # attaque forte + défense forte (def>0 = solide)
    )
    with (model_dir / "team_ratings.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["team", "attack", "defence", "net"])
        for team, att, dfc in ratings:
            w.writerow([team, f"{att:.4f}", f"{dfc:.4f}", f"{att + dfc:.4f}"])

    # 3. Quelques prédictions d'affiche (les 3 meilleures équipes vs les 3 dernières).
    top = [t for t, *_ in ratings[:3]]
    bottom = [t for t, *_ in ratings[-3:]]
    with (model_dir / "sample_predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as fh:
        w = csv.writer(fh)
        w.writerow(["home", "away", "P_home", "P_draw", "P_away"])
        pairs = [(top[0], top[1]), (top[0], bottom[0]),
                 (bottom[0], top[0]), (top[1], top[2])]
        for h, a in pairs:
            pr = predict_1x2(params, h, a)
            w.writerow([h, a, f"{pr['home']:.3f}", f"{pr['draw']:.3f}", f"{pr['away']:.3f}"])

    print("\nTop 5 forces nettes :")
    for team, att, dfc in ratings[:5]:
        print(f"   {team:<16} att={att:+.3f}  def={dfc:+.3f}  net={att + dfc:+.3f}")

    print("\nÉcrit : params.json, team_ratings.csv, sample_predictions.csv")
    return params


if __name__ == "__main__":  # pragma: no cover
    run()
