"""Télécharge l'historique football-data (ligues × saisons du gel) en gzip.

Dossier séparé : data/raw/footballdata_hist/ — ne touche jamais matches.csv.
Idempotent : un fichier déjà présent n'est pas re-téléchargé.

Usage : python validation/etudes/fetch_hist.py
"""
from __future__ import annotations

import gzip
import json
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
FREEZE = Path(__file__).with_name("frozen_hypotheses_hist.json")
OUT = ROOT / "data" / "raw" / "footballdata_hist"
LEAGUES = ["F1", "E0", "D1", "I1", "SP1", "F2", "E1"]
URL = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"


def main() -> None:
    seasons = json.loads(FREEZE.read_text())["saisons_telechargees"]
    OUT.mkdir(parents=True, exist_ok=True)
    ok = skip = miss = 0
    for lg in LEAGUES:
        for s in seasons:
            # Ligue 1 et Premier League : les saisons déjà étudiées ne sont pas téléchargées.
            if lg in ("F1", "E0") and s >= "2122":
                continue
            dest = OUT / f"{lg}_{s}.csv.gz"
            if dest.exists():
                skip += 1
                continue
            try:
                req = Request(URL.format(season=s, league=lg),
                              headers={"User-Agent": "Mozilla/5.0 (etudes historiques)"})
                with urlopen(req, timeout=30) as r:  # noqa: S310
                    data = r.read()
                if len(data) < 500:
                    raise ValueError("fichier vide")
                dest.write_bytes(gzip.compress(data))
                ok += 1
                print(f"{lg} {s}: {len(data)//1024} Ko")
            except (HTTPError, URLError, ValueError, TimeoutError) as e:
                miss += 1
                print(f"{lg} {s}: ABSENT ({e})")
            time.sleep(1)  # politesse envers le serveur
    print(f"\nTéléchargés {ok}, déjà présents {skip}, absents {miss}")


if __name__ == "__main__":
    main()
