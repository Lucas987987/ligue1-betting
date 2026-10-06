"""Études sur l'historique vierge (frozen_hypotheses_hist.json).

1. Couverture Pinnacle par ligue et saison (publiée avant les résultats).
2. Un run confirmatoire des hypothèses gelées, Holm sur la famille.

Usage : python validation/etudes/etudes_hist.py
Sorties : data/validation/hist_couverture.csv, data/validation/etudes_hist.csv,
          résumé markdown sur stdout.
"""
from __future__ import annotations

import gzip
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import etudes_marche as em  # noqa: E402  (mêmes définitions que M1-M7)

ROOT = Path(__file__).resolve().parents[2]
FREEZE = Path(__file__).with_name("frozen_hypotheses_hist.json")
RAW = ROOT / "data" / "raw" / "footballdata_hist"
OUT = ROOT / "data" / "validation"
SEED = 20261006
ISSUES = em.ISSUES
NEED = [f"{p}{c}{i}" for p in ("PS", "B365") for c in ("", "C") for i in ISSUES] + \
       [f"{b}C{i}" for b in em.SOFT_BOOKS for i in ISSUES]


# ---------------------------------------------------------------- chargement
def _read(path: Path) -> pd.DataFrame:
    raw = gzip.decompress(path.read_bytes())
    for enc in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc, on_bad_lines="skip")
        except UnicodeDecodeError:
            continue
    raise ValueError(path)


def load_all() -> tuple[pd.DataFrame, pd.DataFrame]:
    frames, cov = [], []
    for p in sorted(RAW.glob("*.csv.gz")):
        lg, season = p.name.split(".")[0].split("_")
        d = _read(p)
        d = d.dropna(how="all")
        n_tot = len(d)
        d = d.reindex(columns=sorted(set(d.columns) | set(NEED))).copy()
        for c in NEED:
            d[c] = pd.to_numeric(d[c], errors="coerce")
        keep = d.dropna(subset=[f"PS{c}{i}" for c in ("", "C") for i in ISSUES] + ["FTR"])
        keep = keep[keep["FTR"].isin(ISSUES)].copy()
        keep["league"], keep["season"] = lg, season
        dt = pd.to_datetime(keep["Date"], format="%d/%m/%Y", errors="coerce")
        keep["date"] = dt.fillna(pd.to_datetime(keep["Date"], format="%d/%m/%y", errors="coerce"))
        cov.append({"league": lg, "season": season, "matchs": n_tot, "avec_pinnacle_open_close": len(keep),
                    "avec_b365_close": int(keep[[f"B365C{i}" for i in ISSUES]].notna().all(axis=1).sum())})
        frames.append(keep)
    d = pd.concat(frames, ignore_index=True).copy()
    d["mid"] = np.arange(len(d))
    return d.reset_index(drop=True), pd.DataFrame(cov)


def promoted_flags(d: pd.DataFrame) -> pd.DataFrame:
    """Lignes (mid, issue) 'victoire du promu' sur ses 8 premiers matchs."""
    rows = []
    for lg, g in d.groupby("league"):
        seasons = sorted(g["season"].unique())
        teams = {s: set(g.loc[g.season == s, "HomeTeam"]) | set(g.loc[g.season == s, "AwayTeam"])
                 for s in seasons}
        for prev, cur in zip(seasons, seasons[1:]):
            # saisons consécutives uniquement (ex. 1920 -> 2021)
            if int(cur[:2]) != (int(prev[:2]) + 1) % 100:
                continue
            promus = teams[cur] - teams[prev]
            gs = g[g.season == cur].sort_values("date")
            for t in promus:
                m = gs[(gs.HomeTeam == t) | (gs.AwayTeam == t)].head(8)
                for _, r in m.iterrows():
                    rows.append({"mid": r["mid"], "issue": "H" if r["HomeTeam"] == t else "A"})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- stats
def era_gap_boot(L: pd.DataFrame, rng, B: int):
    """Écart EV|move − EV|tout (bet365), bootstrap par match. None si pas de données."""
    if L.empty or (L["dp"] >= MOVE).sum() == 0:
        return None, None
    t = pd.DataFrame({"m": L["mid"].values, "v": L["ev_b365"].values,
                      "w": (L["dp"] >= MOVE).astype(float).values})
    t["vw"] = t["v"] * t["w"]
    g = t.groupby("m")[["v", "vw", "w"]].agg(["sum", "count"])
    s_all, c_all = g[("v", "sum")].values, g[("v", "count")].values
    s_mv, c_mv = g[("vw", "sum")].values, g[("w", "sum")].values
    stat = s_mv.sum() / c_mv.sum() - s_all.sum() / c_all.sum()
    idx = rng.integers(0, len(s_all), size=(B, len(s_all)))
    boots = s_mv[idx].sum(1) / np.maximum(c_mv[idx].sum(1), 1) - s_all[idx].sum(1) / c_all[idx].sum(1)
    return stat, boots


def two_sided(stat, boots):
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return {"stat": stat, "ic_lo": lo, "ic_hi": hi, "p": max(p, 1 / len(boots))}


def one_sided(stat, boots, side):
    r = em.summarize(stat, boots, side)
    return {"stat": r["stat"], "ic_lo": r["ic_lo"], "ic_hi": r["ic_hi"], "p": r["p_un_cote"]}


# ---------------------------------------------------------------- run
def main() -> None:
    global MOVE
    fz = json.loads(FREEZE.read_text())
    MOVE, B = fz["seuil_move"], fz["bootstrap"]
    rng = np.random.default_rng(SEED)
    d, cov = load_all()
    OUT.mkdir(parents=True, exist_ok=True)
    cov.to_csv(OUT / "hist_couverture.csv", index=False)

    L = em.long_format(d)
    L = L.merge(d[["mid", "league", "season"]], on="mid")
    b_o, b_c = em._fair(d, "B365", False), em._fair(d, "B365", True)
    for i in ISSUES:
        m = L["issue"] == i
        L.loc[m, "dp_b365"] = (b_c[i] - b_o[i]).reindex(L.loc[m, "mid"]).values
    res = []

    # R-M4 — prime du nul, Ligue 1
    E = L[L.league == "F1"].pivot(index="mid", columns="issue", values="ev_best").dropna()
    diff = E["D"] - (E["H"] + E["A"]) / 2
    res.append({"id": "R-M4", "n": len(E), **one_sided(diff.mean(), em.cluster_boot(diff, pd.Series(E.index), B, rng), ">"),
                "detail": f"EV_best H={E['H'].mean():.4f} D={E['D'].mean():.4f} A={E['A'].mean():.4f}"})

    # R-M6 — résidu gros move, Premier League (+ secondaire par ligue)
    def resid(sub):
        mv = sub[sub["dp"] >= MOVE]
        r = mv["won"] - mv["p_close"]
        return mv, r, em.cluster_boot(r, mv["mid"], B, rng)
    mv, r, b = resid(L[L.league == "E0"])
    res.append({"id": "R-M6", "n": len(mv), **one_sided(r.mean(), b, ">"),
                "detail": f"fréq={mv['won'].mean():.4f} p_close={mv['p_close'].mean():.4f}"})
    sec = []
    for lg in sorted(L.league.unique()):
        if lg == "E0":
            continue
        mv2, r2, b2 = resid(L[L.league == lg])
        lo, hi = np.percentile(b2, [2.5, 97.5])
        sec.append(f"{lg} {r2.mean():+.3f} [{lo:+.3f};{hi:+.3f}] n={len(mv2)}")

    # H3 — promus
    P = promoted_flags(d)
    if len(P):
        Lp = L.merge(P, on=["mid", "issue"])
        r = Lp["won"] - Lp["p_close"]
        res.append({"id": "H3", "n": len(Lp), **two_sided(r.mean(), em.cluster_boot(r, Lp["mid"], B, rng)),
                    "detail": f"fréq={Lp['won'].mean():.4f} p_close={Lp['p_close'].mean():.4f} (bilatéral)"})

    # H4 — divergence bet365
    Lb = L.dropna(subset=["ev_b365", "dp_b365"])
    dv = Lb[(Lb["dp"] >= MOVE) & (Lb["dp_b365"] <= 0)]
    roi = (dv["won"] * dv["b365c"] - 1).mean()
    res.append({"id": "H4", "n": len(dv), **one_sided(dv["ev_b365"].mean(), em.cluster_boot(dv["ev_b365"], dv["mid"], B, rng), ">"),
                "detail": f"part EV>0={(dv['ev_b365'] > 0).mean():.3f} ROI réalisé={roi:+.4f}"})

    # H5 / H5b — tendance du retard bet365
    for hid, early_mask in (("H5", Lb["season"] <= "1617"),
                            ("H5b", Lb["season"].between("1920", "2122"))):
        late_mask = (Lb["season"] >= "1718") if hid == "H5" else (Lb["season"] >= "2223")
        s_e, b_e = era_gap_boot(Lb[early_mask], rng, B)
        s_l, b_l = era_gap_boot(Lb[late_mask], rng, B)
        if s_e is None or s_l is None:
            res.append({"id": hid, "n": 0, "stat": np.nan, "ic_lo": np.nan, "ic_hi": np.nan, "p": np.nan,
                        "detail": "NON TESTABLE : période sans cote bet365 de clôture"})
            continue
        res.append({"id": hid, "n": int(Lb.loc[early_mask | late_mask, "mid"].nunique()),
                    **one_sided(s_l - s_e, b_l - b_e, "<"),
                    "detail": f"écart tôt={s_e:+.4f} tard={s_l:+.4f}"})

    # H6 — qualité de la ligne du vendredi (descriptif)
    for lg in ("F1", "E0"):
        g = d[d.league == lg]
        po, pc = em._fair(g, "PS", False), em._fair(g, "PS", True)
        y = g["FTR"].values
        lo_ = -np.log(np.array([po.loc[k, r] for k, r in zip(g.index, y)]))
        lc_ = -np.log(np.array([pc.loc[k, r] for k, r in zip(g.index, y)]))
        dl = pd.Series(lo_ - lc_)
        b = em.cluster_boot(dl, pd.Series(np.arange(len(dl))), B, rng)
        res.append({"id": f"H6_{lg}", "n": len(g), **two_sided(dl.mean(), b),
                    "detail": f"LL vendredi={lo_.mean():.5f} clôture={lc_.mean():.5f}"})

    R = pd.DataFrame(res)
    conf = R["id"].isin(["R-M4", "R-M6", "H3", "H4", "H5", "H5b"]) & R["p"].notna()
    p = R.loc[conf, "p"].sort_values()
    m, prev, holm = len(p), 0.0, {}
    for k, (ix, pv) in enumerate(p.items()):
        prev = max(prev, min(1.0, (m - k) * pv))
        holm[ix] = prev
    R["p_holm"] = pd.Series(holm)

    def verdict(r):
        if str(r["detail"]).startswith("NON TESTABLE"):
            return "non testable"
        if r["n"] < fz["n_min"]:
            return "n insuffisant"
        if r["id"].startswith("H6"):
            return "IC contient 0" if r["ic_lo"] <= 0 <= r["ic_hi"] else ("vendredi moins bon" if r["ic_lo"] > 0 else "vendredi meilleur")
        return "CONFIRMÉE" if r["p_holm"] < 0.05 else "NON CONFIRMÉE"
    R["verdict"] = R.apply(verdict, axis=1)
    R.to_csv(OUT / "etudes_hist.csv", index=False, float_format="%.5f")

    print("## Couverture Pinnacle (vendredi + clôture)\n")
    pv = cov.pivot(index="season", columns="league", values="avec_pinnacle_open_close").fillna(0).astype(int)
    print(pv.to_markdown() if hasattr(pv, "to_markdown") else pv.to_string())
    pb = cov.pivot(index="season", columns="league", values="avec_b365_close").fillna(0).astype(int)
    print("\n### dont avec cote bet365 de clôture\n")
    print(pb.to_markdown() if hasattr(pb, "to_markdown") else pb.to_string())
    print(f"\nTotal retenu : {len(d)} matchs\n\n## Résultats (run confirmatoire unique)\n")
    print("| id | n | stat | IC95 | p | p Holm | verdict | détail |\n|---|---|---|---|---|---|---|---|")
    for _, r in R.iterrows():
        ph = "" if pd.isna(r["p_holm"]) else f"{r['p_holm']:.3f}"
        if pd.isna(r["stat"]):
            print(f"| {r['id']} | 0 | | | | | {r['verdict']} | {r['detail']} |")
            continue
        print(f"| {r['id']} | {r['n']} | {r['stat']:+.4f} | [{r['ic_lo']:+.4f}; {r['ic_hi']:+.4f}] | {r['p']:.3f} | {ph} | {r['verdict']} | {r['detail']} |")
    print("\nR-M6 secondaire (autres ligues, non décisionnel) : " + " · ".join(sec))


MOVE = 0.03
if __name__ == "__main__":
    main()
