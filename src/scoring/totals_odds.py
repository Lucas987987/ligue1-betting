"""Extraction des cotes Over/Under (marché 'totals') depuis The Odds API.

Le marché 'totals' a une structure différente du h2h : chaque bookmaker liste des
outcomes nommés "Over" et "Under", chacun avec un champ `point` (le seuil : 2.5,
3.5…) et `price` (la cote). Un même book peut proposer plusieurs seuils.

On regroupe par seuil, et pour chaque (seuil, sens) on retient la MÉDIANE des
cotes sur les books (robuste), en gardant la liste complète pour mesurer la
convergence (fiabilité), exactement comme pour le h2h.

Ne capture que ce que The Odds API fournit : en pratique surtout 2.5. Les seuils
sans cote ne sont pas inventés (pas d'EV possible sans cote).
"""

from __future__ import annotations

import statistics

OVER, UNDER = "Over", "Under"


def extract_totals(event: dict) -> dict:
    """Extrait les cotes over/under par seuil d'un événement Odds API.

    Retourne : { seuil(float) : {
        'over': cote_médiane, 'under': cote_médiane,
        'over_prices': [...], 'under_prices': [...]  # par book, pour convergence
    } }
    Seuls les seuils ayant À LA FOIS over et under cotés sont inclus.
    """
    # Collecte brute : prices[seuil][sens] = liste de cotes sur les books.
    prices: dict[float, dict[str, list[float]]] = {}
    for bk in event.get("bookmakers", []):
        for mk in bk.get("markets", []):
            if mk.get("key") != "totals":
                continue
            for oc in mk.get("outcomes", []):
                name = oc.get("name")
                point = oc.get("point")
                price = oc.get("price")
                if name not in (OVER, UNDER) or point is None or price is None:
                    continue
                if price <= 1.0:
                    continue
                seuil = float(point)
                slot = prices.setdefault(seuil, {"over": [], "under": []})
                slot["over" if name == OVER else "under"].append(float(price))

    # Consolidation : médiane par sens, seuils complets uniquement.
    out = {}
    for seuil, sides in prices.items():
        if not sides["over"] or not sides["under"]:
            continue  # seuil incomplet (manque over ou under) → on l'écarte
        out[seuil] = {
            "over": statistics.median(sides["over"]),
            "under": statistics.median(sides["under"]),
            "over_prices": sides["over"],
            "under_prices": sides["under"],
        }
    return out
