"""Validation walk-forward du modèle enrichi de la FORME.

Compare le modèle-forme au modèle de base et au marché, sur exactement les mêmes
matchs, même protocole sans fuite vers le futur. Verdict : la forme-résidu
améliore-t-elle la log-loss (réf. bayésien 1.0009, fréquentiste 1.0037,
marché 0.9812) — ou son coefficient ressort-il négligeable ?

ANTI-FUITE (le point sensible, documenté clairement) :
  À chaque date d'évaluation D :
    1. On entraîne UN modèle de base sur le passé strict (matchs < D).
    2. On l'utilise pour calculer la forme-résidu de chaque équipe à partir de
       ses matchs récents (tous < D). Compromis assumé : ces résidus emploient un
       modèle qui a vu ces matchs (léger optimisme), MAIS aucune information ≥ D
       n'entre. La fuite vers le FUTUR — la seule qui invaliderait la validation —
       est exclue. C'est la pratique standard.
    3. On calcule la forme de CHAQUE match d'entraînement (avec ce même modèle de
       base) pour pouvoir réajuster un modèle ENRICHI sur le passé.
    4. On prédit le match de D avec le modèle enrichi, en utilisant la forme des
       deux équipes calculée à l'étape 2.

Coût : un fit de base + un fit enrichi par date. Réaliste pour MAP fréquentiste.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

from model.dixoncoles import fit as fit_base
from model.dixoncoles_form import fit_form, predict_1x2_form
from signals.form import compute_form, expected_goals_from_params, MatchOutcome

from validation.walkforward import (
    BURN_IN_SEASONS, DEFAULT_MATCHES, DEFAULT_OUT, EPS,
    Metrics, calibration_table, load_matches, market_probs,
)

FORM_WINDOW = 5


def _expected_goals_factory(base_params):
    """Fournit expected_goals(date, home, away) à partir d'un modèle de base.
    La date n'est pas utilisée (le modèle de base est déjà figé sur le passé de
    la date d'évaluation) — signature conservée pour l'API de compute_form."""
    known = set(base_params.teams)

    def eg(_date, home, away):
        if home not in known or away not in known:
            return 1.3, 1.1  # valeurs neutres si équipe inconnue (rare)
        return expected_goals_from_params(base_params, home, away)
    return eg


def walk_forward_form(matches):
    m_form, m_market = Metrics(), Metrics()
    n_unknown = 0
    gammas, deltas = [], []

    eval_matches = [m for m in matches if m.season not in BURN_IN_SEASONS]
    eval_dates = sorted({m.date for m in eval_matches})

    # Historique au format MatchOutcome (pour compute_form).
    all_outcomes = [
        MatchOutcome(m.date, m.home, m.away, m.hg, m.ag) for m in matches
    ]

    for d in eval_dates:
        train = [m for m in matches if m.date < d]
        if len(train) < 150:
            continue

        # 1. Modèle de base sur le passé strict.
        base = fit_base([m.home for m in train], [m.away for m in train],
                        [m.hg for m in train], [m.ag for m in train])
        known = set(base.teams)
        eg = _expected_goals_factory(base)

        hist_before_d = [o for o in all_outcomes if o.date < d]

        # 3. Forme de chaque match d'entraînement (pour réajuster enrichi).
        foh, fdh, foa, fda = [], [], [], []
        for m in train:
            fh = compute_form(m.date, m.home, hist_before_d, eg, FORM_WINDOW)
            fa = compute_form(m.date, m.away, hist_before_d, eg, FORM_WINDOW)
            foh.append(fh.off); fdh.append(fh.deff)
            foa.append(fa.off); fda.append(fa.deff)

        enriched = fit_form(
            [m.home for m in train], [m.away for m in train],
            [m.hg for m in train], [m.ag for m in train],
            foh, fdh, foa, fda,
        )
        gammas.append(enriched.gamma); deltas.append(enriched.delta)

        # 4. Prédire les matchs de D avec la forme des deux équipes.
        for m in (mt for mt in eval_matches if mt.date == d):
            if m.home not in known or m.away not in known:
                n_unknown += 1
                continue
            fh = compute_form(d, m.home, hist_before_d, eg, FORM_WINDOW)
            fa = compute_form(d, m.away, hist_before_d, eg, FORM_WINDOW)
            probs = predict_1x2_form(
                enriched, m.home, m.away,
                form_off_home=fh.off, form_def_home=fh.deff,
                form_off_away=fa.off, form_def_away=fa.deff,
            )
            m_form.add(probs, m.result)
            if m.odds is not None:
                m_market.add(market_probs(m.odds), m.result)

    return m_form, m_market, n_unknown, gammas, deltas


def run(matches_path: Path = DEFAULT_MATCHES, out_dir: Path = DEFAULT_OUT):
    matches = load_matches(matches_path)
    print(f"Matchs chargés : {len(matches)}")

    m_form, m_market, n_unknown, gammas, deltas = walk_forward_form(matches)

    avg_g = sum(gammas) / len(gammas) if gammas else float("nan")
    avg_d = sum(deltas) / len(deltas) if deltas else float("nan")

    print(f"\nMatchs évalués : {m_form.n}  (ignorés : {n_unknown})")
    print(f"\nCoefficients de forme estimés (moyenne sur les fits) :")
    print(f"  γ (offensive) : {avg_g:+.4f}")
    print(f"  δ (défensive) : {avg_d:+.4f}")

    print("\n=== LOG-LOSS ===")
    print(f"  Modèle + forme : {m_form.log_loss:.4f}")
    print(f"  (réf. bayésien : 1.0009 · fréquentiste : 1.0037)")
    print(f"  Marché         : {m_market.log_loss:.4f}")

    print("\n=== BRIER ===")
    print(f"  Modèle + forme : {m_form.brier:.4f}  (réf. bayésien 0.5981)")

    print("\n=== VERDICT ===")
    if abs(avg_g) < 0.03 and abs(avg_d) < 0.03:
        print("  ~ Coefficients de forme quasi nuls : la forme-résidu n'apporte")
        print("    presque rien. Le modèle de base suffit. Résultat honnête.")
    else:
        print(f"  Coefficients non négligeables (γ={avg_g:+.3f}, δ={avg_d:+.3f}).")
        print(f"  Comparer la log-loss {m_form.log_loss:.4f} à la référence")
        print(f"  bayésienne 1.0009 : meilleure => la forme aide ; pire ou égale")
        print(f"  => elle ajoute du bruit malgré un coefficient non nul.")

    print("\n=== CALIBRATION ===")
    print("  tranche      n     prédit   réel")
    for label, k, pred, real in calibration_table(m_form):
        flag = "" if abs(pred - real) < 0.05 else "  <-- écart"
        print(f"  {label:<10} {k:>4}   {pred:.3f}   {real:.3f}{flag}")

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "summary_form.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "model_form", "ref_bayes", "market"])
        w.writerow(["log_loss", f"{m_form.log_loss:.4f}", "1.0009", f"{m_market.log_loss:.4f}"])
        w.writerow(["brier", f"{m_form.brier:.4f}", "0.5981", f"{m_market.brier:.4f}"])
        w.writerow(["gamma_mean", f"{avg_g:.4f}", "", ""])
        w.writerow(["delta_mean", f"{avg_d:.4f}", "", ""])
        w.writerow(["n_evaluated", m_form.n, "", m_market.n])

    print("\nÉcrit : validation/summary_form.csv")
    return m_form, m_market, avg_g, avg_d


if __name__ == "__main__":  # pragma: no cover
    run()
