"""CLV Over/Under — juge de paix des paris sur les totaux.

Compare la cote du pari over/under (journal bet_log_ou.csv, cote précoce) à la
clôture du marché. LIMITE des données : football-data ne fournit la clôture que
pour le seuil 2.5 (P>2.5 / P<2.5 = Pinnacle ; Avg>2.5 / Avg<2.5 = moyenne marché).

Conséquence assumée :
  - Paris sur 2.5  → CLV calculable (brut + dévigorisé).
  - Paris sur 0.5/1.5/3.5 → AUCUNE clôture de référence → statut 'no_reference'.
    On ne devine pas, on ne calcule pas. C'est une limite des données.

Même logique que clv.py (1/N/2) : jointure via date+équipes, repli moyenne marché
si Pinnacle manque, CLV brut = cote_pari/cote_clôture − 1.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from common.teams import TeamResolver, UnknownTeamError
from scoring.betlog_ou import OUBetRecord, load_log_ou, DEFAULT_LOG_OU

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MATCHES = _ROOT / "data" / "processed" / "matches.csv"
DEFAULT_OUT = _ROOT / "data" / "bets" / "clv_report_ou.csv"

# Clôture over/under 2.5 dans football-data (seul seuil disponible).
CLOSE_PINNACLE = {"over": "P>2.5", "under": "P<2.5"}
CLOSE_MARKET = {"over": "Avg>2.5", "under": "Avg<2.5"}

SUPPORTED_THRESHOLD = 2.5


@dataclass
class ClvOUResult:
    bet: OUBetRecord
    close_odds: float | None
    close_source: str            # 'pinnacle' | 'market' | 'none'
    clv_raw: float | None
    clv_devig: float | None
    status: str                  # 'matched' | 'pending' | 'no_reference' | 'unresolved'


def _parse_date(s: str) -> datetime | None:
    for fmt in ("%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d"):
        try:
            return datetime.strptime(s.strip(), fmt)
        except ValueError:
            continue
    return None


def load_closings(matches_path: Path):
    closings = {}
    with Path(matches_path).open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            h = (row.get("HomeTeamCanonical") or "").strip()
            a = (row.get("AwayTeamCanonical") or "").strip()
            d = _parse_date(row.get("Date", ""))
            if h and a and d:
                closings[(d.strftime("%Y-%m-%d"), h, a)] = row
    return closings


def _read_close(row, side: str) -> tuple[float | None, str]:
    """Cote de clôture over/under 2.5 : Pinnacle d'abord, moyenne en repli."""
    for cols, src in ((CLOSE_PINNACLE, "pinnacle"), (CLOSE_MARKET, "market")):
        v = (row.get(cols[side]) or "").strip()
        if v:
            try:
                f = float(v)
                if f > 1.0:
                    return f, src
            except ValueError:
                pass
    return None, "none"


def _all_close(row, cols) -> dict | None:
    out = {}
    for side, col in cols.items():
        v = (row.get(col) or "").strip()
        try:
            f = float(v)
        except (ValueError, TypeError):
            return None
        if f <= 1.0:
            return None
        out[side] = f
    return out


def _devig(odds: dict[str, float]) -> dict[str, float]:
    inv = {k: 1.0 / v for k, v in odds.items()}
    s = sum(inv.values())
    return {k: v / s for k, v in inv.items()}


def compute_clv_for_bet(bet: OUBetRecord, closings: dict, resolver: TeamResolver) -> ClvOUResult:
    # Seuls les paris sur 2.5 ont une clôture de référence.
    if abs(bet.threshold - SUPPORTED_THRESHOLD) > 1e-9:
        return ClvOUResult(bet, None, "none", None, None, "no_reference")

    md = _parse_date(bet.match_date)
    if md is None:
        return ClvOUResult(bet, None, "none", None, None, "pending")
    key = (md.strftime("%Y-%m-%d"), bet.home, bet.away)
    row = closings.get(key)
    if row is None:
        try:
            hc = resolver.to_canonical(bet.home, "oddsapi")
            ac = resolver.to_canonical(bet.away, "oddsapi")
            row = closings.get((md.strftime("%Y-%m-%d"), hc, ac))
        except UnknownTeamError:
            row = None
    if row is None:
        return ClvOUResult(bet, None, "none", None, None, "pending")

    close_odds, src = _read_close(row, bet.side)
    if close_odds is None:
        return ClvOUResult(bet, None, "none", None, None, "unresolved")

    clv_raw = bet.odds / close_odds - 1.0

    clv_devig = None
    cols = CLOSE_PINNACLE if src == "pinnacle" else CLOSE_MARKET
    close_all = _all_close(row, cols)
    if close_all is not None:
        p_close = _devig(close_all)[bet.side]
        p_bet = 1.0 / bet.odds
        if p_bet > 0:
            clv_devig = p_close / p_bet - 1.0

    return ClvOUResult(bet, close_odds, src, clv_raw, clv_devig, "matched")


def run(
    log_path: Path = DEFAULT_LOG_OU,
    matches_path: Path = DEFAULT_MATCHES,
    out_path: Path = DEFAULT_OUT,
    resolver: TeamResolver | None = None,
):
    bets = load_log_ou(log_path)
    print(f"Paris over/under journalisés : {len(bets)}")
    if not bets:
        print("Aucun pari over/under à évaluer (normal hors saison).")
        return []

    closings = load_closings(matches_path)
    resolver = resolver or TeamResolver.from_csv()
    results = [compute_clv_for_bet(b, closings, resolver) for b in bets]

    matched = [r for r in results if r.status == "matched"]
    pending = [r for r in results if r.status == "pending"]
    no_ref = [r for r in results if r.status == "no_reference"]

    print(f"Appariés (seuil 2.5, match joué) : {len(matched)}")
    print(f"En attente (match pas encore joué/ingéré) : {len(pending)}")
    print(f"Sans référence (seuil ≠ 2.5, pas de clôture) : {len(no_ref)}")

    if matched:
        raw = [r.clv_raw for r in matched if r.clv_raw is not None]
        dv = [r.clv_devig for r in matched if r.clv_devig is not None]
        avg_raw = sum(raw) / len(raw) if raw else float("nan")
        pos = sum(1 for v in raw if v > 0)
        print("\n=== CLV OVER/UNDER (paris 2.5 appariés) ===")
        print(f"  CLV brut moyen       : {avg_raw:+.4f} ({avg_raw*100:+.2f} %)")
        if dv:
            print(f"  CLV dévigorisé moyen : {sum(dv)/len(dv):+.4f}")
        print(f"  Paris battant la clôture : {pos}/{len(raw)} "
              f"({100*pos/len(raw):.0f} %)")
        print("\n=== VERDICT ===")
        if avg_raw > 0:
            print("  ✓ CLV moyen POSITIF sur l'over/under 2.5 : le scoring obtient")
            print("    de meilleures cotes que la clôture. Signe encourageant.")
        else:
            print("  ✗ CLV moyen négatif : pas d'avantage sur l'over/under 2.5.")
        print(f"\n  ⚠ Échantillon : {len(raw)} paris. Non fiable avant ~100-200.")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(out_path).open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["match_date", "home", "away", "threshold", "side", "bet_odds",
                    "close_odds", "close_source", "clv_raw", "clv_devig", "status"])
        for r in results:
            w.writerow([
                r.bet.match_date, r.bet.home, r.bet.away, r.bet.threshold, r.bet.side,
                f"{r.bet.odds:.4f}",
                f"{r.close_odds:.4f}" if r.close_odds is not None else "",
                r.close_source,
                f"{r.clv_raw:.4f}" if r.clv_raw is not None else "",
                f"{r.clv_devig:.4f}" if r.clv_devig is not None else "",
                r.status,
            ])
    print(f"\nÉcrit : {out_path}")
    return results


if __name__ == "__main__":  # pragma: no cover
    run()
