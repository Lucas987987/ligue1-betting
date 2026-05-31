"""Choix de l'hyperparamètre ξ (décroissance temporelle) — sans tricher.

Décroissance Dixon-Coles : chaque match passé pèse φ(t) = exp(-ξ·Δt_années)
dans l'estimation. ξ=0 = pas d'oubli (modèle actuel) ; ξ grand = seul le passé
récent compte.

ξ est un HYPERPARAMÈTRE : on ne l'estime pas dans le modèle, on le choisit par
validation. Pour ne pas surajuster ξ aux données d'évaluation (cf. ARCHITECTURE.md
§3, découpage 3 blocs), on sépare :

  - SÉLECTION de ξ : walk-forward évalué sur les saisons 2324 + 2425.
    On garde le ξ qui minimise la log-loss sur ce bloc.
  - TEST final : on évalue ce ξ (et ξ=0 pour comparaison) sur 2526 UNIQUEMENT,
    une seule fois. Ce chiffre est l'estimation honnête de performance.

La saison 2526 ne doit influencer AUCUNE décision de sélection. C'est la règle.
"""

from __future__ import annotations

import csv
import math
from datetime import datetime
from pathlib import Path

import numpy as np

from model.dixoncoles import fit, predict_1x2

_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATCHES = _ROOT / "data" / "processed" / "matches.csv"
DEFAULT_OUT = _ROOT / "data" / "validation"

HOME, AWAY = "HomeTeamCanonical", "AwayTeamCanonical"
HG, AG, RES = "FTHG", "FTAG", "FTR"

BURN_IN = {"2122", "2223"}        # amorçage : entraînement seul
VALID_SEASONS = {"2324", "2425"}  # bloc de SÉLECTION de ξ
TEST_SEASON = "2526"              # bloc de TEST final, intouché

# Grille de ξ à tester (par an). 0 = pas de décroissance.
XI_GRID = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.80, 1.20]

EPS = 1e-15


def _parse_date(s: str) -> datetime:
    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            return datetime.strptime(s.strip(), fmt)
        except ValueError:
            continue
    raise ValueError(s)


def load(path: Path):
    rows = []
    with path.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            h, a = (r.get(HOME) or "").strip(), (r.get(AWAY) or "").strip()
            gh, ga = (r.get(HG) or "").strip(), (r.get(AG) or "").strip()
            res = (r.get(RES) or "").strip()
            if not (h and a and gh and ga and res in "HDA" and len(res) == 1):
                continue
            try:
                rows.append({
                    "date": _parse_date(r["Date"]),
                    "season": (r.get("season") or "").strip(),
                    "home": h, "away": a,
                    "hg": int(float(gh)), "ag": int(float(ga)), "res": res,
                })
            except (ValueError, KeyError):
                continue
    rows.sort(key=lambda m: m["date"])
    return rows


def _log_loss_on(matches, eval_seasons, xi) -> tuple[float, int]:
    """Walk-forward : pour chaque date des saisons à évaluer, entraîne sur le
    passé (pondéré par ξ), prédit, accumule la log-loss."""
    eval_rows = [m for m in matches if m["season"] in eval_seasons]
    eval_dates = sorted({m["date"] for m in eval_rows})
    total_ll, n = 0.0, 0

    for d in eval_dates:
        train = [m for m in matches if m["date"] < d]
        if len(train) < 100:
            continue
        # Poids temporels : Δt en années avant la date de prédiction.
        if xi > 0:
            dt_years = np.array([(d - m["date"]).days / 365.25 for m in train])
            weights = np.exp(-xi * dt_years)
        else:
            weights = None
        params = fit(
            [m["home"] for m in train], [m["away"] for m in train],
            [m["hg"] for m in train], [m["ag"] for m in train],
            weights=weights,
        )
        known = set(params.teams)
        for m in (mt for mt in eval_rows if mt["date"] == d):
            if m["home"] not in known or m["away"] not in known:
                continue
            p = predict_1x2(params, m["home"], m["away"])
            key = {"H": "home", "D": "draw", "A": "away"}[m["res"]]
            total_ll += -math.log(min(max(p[key], EPS), 1 - EPS))
            n += 1
    return (total_ll / n if n else float("nan")), n


def run(matches_path: Path = DEFAULT_MATCHES, out_dir: Path = DEFAULT_OUT):
    matches = load(matches_path)
    print(f"Matchs chargés : {len(matches)}")

    # --- ÉTAPE 1 : sélection de ξ sur le bloc validation (2324+2425) ---
    print(f"\n=== SÉLECTION de ξ sur {sorted(VALID_SEASONS)} ===")
    print("   ξ        log-loss(valid)   n")
    results = []
    for xi in XI_GRID:
        ll, n = _log_loss_on(matches, VALID_SEASONS, xi)
        results.append((xi, ll, n))
        print(f"   {xi:<6.2f}   {ll:.4f}            {n}")

    best_xi, best_ll, _ = min(results, key=lambda t: t[1])
    print(f"\n   → ξ retenu : {best_xi} (log-loss validation {best_ll:.4f})")

    # --- ÉTAPE 2 : TEST final sur 2526, une seule fois ---
    # On compare ξ=0 (baseline) et le ξ retenu, sur la saison jamais touchée.
    print(f"\n=== TEST final sur {TEST_SEASON} (intouché) ===")
    ll_test_0, n_test = _log_loss_on(matches, {TEST_SEASON}, 0.0)
    ll_test_best, _ = _log_loss_on(matches, {TEST_SEASON}, best_xi)
    print(f"   ξ=0 (sans décroissance) : log-loss {ll_test_0:.4f}  (n={n_test})")
    print(f"   ξ={best_xi} (retenu)        : log-loss {ll_test_best:.4f}")

    gain = ll_test_0 - ll_test_best
    print("\n=== VERDICT ===")
    if best_xi == 0.0:
        print("   La décroissance temporelle n'aide pas : ξ=0 est optimal en validation.")
        print("   Le modèle non pondéré reste le meilleur choix.")
    elif gain > 0:
        print(f"   ✓ La décroissance ξ={best_xi} AMÉLIORE la log-loss test de {gain:+.4f}.")
        print(f"     À intégrer au modèle de production.")
    else:
        print(f"   ~ ξ={best_xi} meilleur en validation mais PAS en test ({gain:+.4f}).")
        print(f"     Signe d'un léger surajustement de ξ. Prudence : garder ξ=0 ou tester plus.")

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "xi_tuning.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["xi", "logloss_validation", "n_validation"])
        for xi, ll, n in results:
            w.writerow([xi, f"{ll:.4f}", n])
        w.writerow([])
        w.writerow(["best_xi", best_xi, ""])
        w.writerow(["logloss_test_xi0", f"{ll_test_0:.4f}", n_test])
        w.writerow([f"logloss_test_xi{best_xi}", f"{ll_test_best:.4f}", ""])

    print("\nÉcrit : validation/xi_tuning.csv")
    return best_xi, ll_test_0, ll_test_best


if __name__ == "__main__":  # pragma: no cover
    run()
