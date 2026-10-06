"""Mesure forward F1/F2/F3 sur les exports du Worker (protocole_forward.json).

Lit data/raw/d1/{prices,closing,results}_*.csv.gz, applique le protocole gelé,
écrit :
  data/validation/forward_entries.csv   (une ligne par entrée F2)
  data/validation/forward_latence.csv   (F1, une ligne par détection × book)
  data/validation/forward_verdicts.json (verdicts confirmatoires : écrits UNE fois)
et un résumé markdown sur stdout.

Usage : python validation/etudes/forward_fr.py
"""
from __future__ import annotations

import glob
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROTO = json.loads(Path(__file__).with_name("protocole_forward.json").read_text())
D1 = ROOT / "data" / "raw" / "d1"
OUT = ROOT / "data" / "validation"
FR = PROTO["books_fr"]
WIN = pd.Timedelta(hours=6)
MOVE = 0.03
B = 5000
SEED = 20261006


# ------------------------------------------------------------------ lecture
def _read_many(pattern: str) -> pd.DataFrame:
    frames = []
    for p in sorted(glob.glob(str(D1 / pattern))):
        try:
            f = pd.read_csv(p)
        except pd.errors.EmptyDataError:
            continue
        if len(f):
            f["_file"] = Path(p).name
            frames.append(f)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _issue(df: pd.DataFrame) -> pd.Series:
    return np.select([df["outcome"] == df["home"], df["outcome"] == df["away"],
                      df["outcome"] == "Draw"], ["H", "A", "D"], default="?")


def load():
    P = _read_many("prices_*.csv.gz")
    C = _read_many("closing_*.csv.gz")
    R = _read_many("results_*.csv.gz")
    if P.empty:
        return None, None, None
    P = P[P["market"] == "h2h"].drop(columns="_file").drop_duplicates()
    P["issue"] = _issue(P)
    P = P[P["issue"] != "?"]
    for c in ("ts", "commence_time"):
        P[c] = pd.to_datetime(P[c], utc=True)
    P = P[P["ts"] < P["commence_time"]].sort_values("ts")
    if not C.empty:
        C = C[C["market"] == "h2h"].copy()
        C["issue"] = _issue(C)
        C = C.drop_duplicates(["event_id", "book", "issue"], keep="last")  # export le plus récent
    if not R.empty:
        R = R.drop_duplicates("event_id", keep="last")
        R["res"] = np.select([R.home_score > R.away_score, R.home_score < R.away_score], ["H", "A"], "D")
    return P, C, R


# ---------------------------------------------------------------- séries
def fair_series(df: pd.DataFrame) -> pd.DataFrame:
    """Probas dévigorisées d'un book au fil des changements (report du dernier prix)."""
    w = df.pivot_table(index="ts", columns="issue", values="price", aggfunc="last").sort_index().ffill()
    w = w.dropna(subset=[c for c in ("H", "D", "A") if c in w.columns])
    if not all(c in w.columns for c in ("H", "D", "A")) or w.empty:
        return pd.DataFrame()
    inv = 1 / w[["H", "D", "A"]]
    return inv.div(inv.sum(axis=1), axis=0)


def price_at(df: pd.DataFrame, issue: str, t) -> float | None:
    x = df[(df["issue"] == issue) & (df["ts"] <= t)]
    return None if x.empty else float(x["price"].iloc[-1])


def detections(pin: pd.DataFrame) -> list[dict]:
    out = []
    for i in ("H", "D", "A"):
        s = pin[i]
        for t, p in s.items():
            # Fenêtre [t−6 h, t[ en « prix courant » : on inclut la valeur en vigueur
            # à t−6 h (dernier changement avant la fenêtre), puisque les exports
            # ne stockent que les changements.
            before = s[s.index < t - WIN]
            past = s[(s.index >= t - WIN) & (s.index < t)]
            if len(before):
                past = pd.concat([before.iloc[-1:], past])
            if past.empty:
                continue
            if p - past.min() >= MOVE:
                out.append({"issue": i, "t": t, "p_t": p, "p_min": past.min(), "t_min": past.idxmin()})
                break  # première détection seulement
    return out


def boot_mean(v: np.ndarray, clusters: np.ndarray, rng) -> tuple[float, float]:
    df = pd.DataFrame({"v": v, "c": clusters}).groupby("c")["v"].agg(["sum", "count"])
    s, c = df["sum"].values, df["count"].values
    idx = rng.integers(0, len(s), size=(B, len(s)))
    bm = s[idx].sum(1) / c[idx].sum(1)
    return tuple(np.percentile(bm, [2.5, 97.5]))


# ---------------------------------------------------------------- main
def main() -> None:
    rng = np.random.default_rng(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    P, C, R = load()
    print(f"## Forward books FR — {datetime.now(timezone.utc):%Y-%m-%d}\n")
    if P is None:
        print("Aucun export de prix. Rien à mesurer.")
        return

    # Clôtures Pinnacle + filtre qualité
    close_fair, excl = {}, 0
    if not C.empty:
        pc = C[C["book"] == "pinnacle"]
        for ev, g in pc.groupby("event_id"):
            if g["last_pass_min_before"].max() > 15 or g["issue"].nunique() < 3:
                excl += 1
                continue
            inv = {r.issue: 1 / r.price for r in g.itertuples()}
            s = sum(inv.values())
            close_fair[ev] = {k: v / s for k, v in inv.items()}
    res = {} if R.empty else dict(zip(R["event_id"], R["res"]))

    entries, lat = [], []
    for ev, g in P.groupby("event_id"):
        pin = fair_series(g[g["book"] == "pinnacle"])
        if pin.empty:
            continue
        for d in detections(pin):
            for bk in FR:
                gb = g[g["book"] == bk]
                if gb.empty:
                    continue
                # F1 — latence
                fb = fair_series(gb)
                if not fb.empty:
                    base_rows = fb[fb.index <= d["t_min"]]
                    base = base_rows[d["issue"]].iloc[-1] if len(base_rows) else fb[d["issue"]].iloc[0]
                    target = base + 0.5 * (d["p_t"] - d["p_min"])
                    after = fb[(fb.index >= d["t"]) & (fb[d["issue"]] >= target)]
                    before = fb[fb.index <= d["t"]]
                    if len(before) and before[d["issue"]].iloc[-1] >= target:
                        delay, followed = 0.0, True
                    elif len(after):
                        delay, followed = (after.index[0] - d["t"]).total_seconds() / 60, True
                    else:
                        delay, followed = np.nan, False
                    lat.append({"event_id": ev, "issue": d["issue"], "book": bk,
                                "delai_min": delay, "suivi": followed})
                # F2 — entrée
                pr = price_at(gb, d["issue"], d["t"])
                if pr is None:
                    continue
                ev_det = pr * d["p_t"] - 1
                if ev_det <= 0:
                    continue
                cf = close_fair.get(ev)
                won = res.get(ev)
                entries.append({
                    "event_id": ev, "home": g["home"].iloc[0], "away": g["away"].iloc[0],
                    "commence_time": g["commence_time"].iloc[0], "issue": d["issue"], "book": bk,
                    "t_detection": d["t"], "cote": pr, "p_pin_t": d["p_t"], "ev_detection": ev_det,
                    "clv": None if cf is None else pr * cf[d["issue"]] - 1,
                    "gain": None if won is None else (pr - 1 if won == d["issue"] else -1.0),
                })

    E = pd.DataFrame(entries)
    Lt = pd.DataFrame(lat)
    E.to_csv(OUT / "forward_entries.csv", index=False)
    Lt.to_csv(OUT / "forward_latence.csv", index=False)

    n_ev = P["event_id"].nunique()
    print(f"Matchs capturés : {n_ev} · clôtures Pinnacle fiables : {len(close_fair)} · exclues (>15 min) : {excl}\n")

    # F1
    print("### F1 — latence des books FR après un mouvement Pinnacle ≥ 3 pts\n")
    if Lt.empty:
        print("Aucune détection pour l'instant.\n")
    else:
        print("| book | détections | suivis avant coup d'envoi | délai médian (min) |\n|---|---|---|---|")
        for bk, g in Lt.groupby("book"):
            print(f"| {bk} | {len(g)} | {g['suivi'].mean():.0%} | {g['delai_min'].median():.0f} |")
        print()

    # F2
    verdict_path = OUT / "forward_verdicts.json"
    verdicts = json.loads(verdict_path.read_text()) if verdict_path.exists() else {}
    print("### F2 — entrées FR au-dessus de la juste Pinnacle à la détection\n")
    Ec = E.dropna(subset=["clv"]) if not E.empty else E
    if Ec.empty:
        print("Aucune entrée mesurable (pas encore de clôture fiable).\n")
    else:
        lo, hi = boot_mean(Ec["clv"].values, Ec["event_id"].values, rng)
        n, nm = len(Ec), Ec["event_id"].nunique()
        Er = Ec.dropna(subset=["gain"])
        roi = Er["gain"].mean() if len(Er) else float("nan")
        print(f"n = {n} entrées sur {nm} matchs · CLV moyen {Ec['clv'].mean():+.2%} [IC95 {lo:+.2%} ; {hi:+.2%}] "
              f"· CLV>0 : {(Ec['clv'] > 0).mean():.0%} · ROI {roi:+.2%} (n = {len(Er)})\n")
        print("| book | n | CLV moyen | ROI |\n|---|---|---|---|")
        for bk, g in Ec.groupby("book"):
            gr = g.dropna(subset=["gain"])
            print(f"| {bk} | {len(g)} | {g['clv'].mean():+.2%} | {gr['gain'].mean() if len(gr) else float('nan'):+.2%} |")
        print()
        if "F2" not in verdicts and n >= 30 and nm >= 30:
            verdicts["F2"] = {"date": f"{datetime.now(timezone.utc):%Y-%m-%d}", "n": n, "matchs": nm,
                              "clv": Ec["clv"].mean(), "ic95": [lo, hi],
                              "verdict": "CONFIRMÉE" if lo > 0 else "NON CONFIRMÉE",
                              "roi": roi, "n_roi": len(Er)}
            print(f"**Verdict F2 enregistré (définitif) : {verdicts['F2']['verdict']}**\n")
        elif "F2" in verdicts:
            print(f"Verdict F2 déjà rendu le {verdicts['F2']['date']} : {verdicts['F2']['verdict']} "
                  f"(les chiffres ci-dessus sont descriptifs)\n")
        else:
            print(f"Verdict F2 : en attente (n ≥ 30 entrées et ≥ 30 matchs requis)\n")
        if len(Er) >= 100:
            print(f"ROI sur n ≥ 100 : {roi:+.2%} → {'condition de mise remplie' if roi > 0 and verdicts.get('F2', {}).get('verdict') == 'CONFIRMÉE' else 'pas de mise'}\n")

    # F3
    print("### F3 — prime du nul chez les books FR à la clôture\n")
    rows = []
    if not C.empty:
        cf_ = C[C["book"].isin(FR)]
        for ev, g in cf_.groupby("event_id"):
            if ev not in close_fair:
                continue
            best = g.groupby("issue")["price"].max()
            if set(best.index) >= {"H", "D", "A"}:
                e = {i: best[i] * close_fair[ev][i] - 1 for i in ("H", "D", "A")}
                rows.append({"event_id": ev, "diff": e["D"] - (e["H"] + e["A"]) / 2, **e})
    F3 = pd.DataFrame(rows)
    if F3.empty:
        print("Aucune clôture FR mesurable pour l'instant.\n")
    else:
        lo, hi = boot_mean(F3["diff"].values, F3["event_id"].values, rng)
        print(f"n = {len(F3)} matchs · EV_best FR : 1 {F3['H'].mean():+.2%} · N {F3['D'].mean():+.2%} · 2 {F3['A'].mean():+.2%} "
              f"· prime du nul {F3['diff'].mean():+.2%} [IC95 {lo:+.2%} ; {hi:+.2%}]\n")
        if "F3" not in verdicts and len(F3) >= 30:
            verdicts["F3"] = {"date": f"{datetime.now(timezone.utc):%Y-%m-%d}", "n": len(F3),
                              "stat": F3["diff"].mean(), "ic95": [lo, hi],
                              "verdict": "CONFIRMÉE" if lo > 0 else "NON CONFIRMÉE"}
            print(f"**Verdict F3 enregistré (définitif) : {verdicts['F3']['verdict']}**\n")
        elif "F3" in verdicts:
            print(f"Verdict F3 déjà rendu le {verdicts['F3']['date']} : {verdicts['F3']['verdict']}\n")
        else:
            print("Verdict F3 : en attente (n ≥ 30 matchs requis)\n")

    verdict_path.write_text(json.dumps(verdicts, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
