"""Tests d'ingestion football-data — garantit l'immutabilité de la couche raw.

Utilise un downloader injecté (faux serveur) : aucun accès réseau requis.

Lancer depuis la racine :  PYTHONPATH=src python -m pytest tests/
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from ingestion.footballdata import (
    RawImmutabilityError,
    _capture_name,
    ingest_recent,
    ingest_season,
    recent_seasons,
    season_url,
)

UTC = timezone.utc
FAKE_V1 = b"Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\nF1,15/08/25,Rennes,Marseille,2,1,H\n"
FAKE_V2 = FAKE_V1 + b"F1,16/08/25,Lens,Lyon,0,0,D\n"


# --------------------------------------------------------------------- #
# Calcul des saisons
# --------------------------------------------------------------------- #
def test_cinq_saisons_format_et_ordre():
    s = recent_seasons(5, today=datetime(2026, 5, 31, tzinfo=UTC))
    assert s == ["2122", "2223", "2324", "2425", "2526"]


def test_bascule_de_saison_en_juillet():
    # Avant juillet : saison courante = année-1/année
    assert recent_seasons(1, today=datetime(2026, 5, 31, tzinfo=UTC))[-1] == "2526"
    # À partir d'août : nouvelle saison
    assert recent_seasons(1, today=datetime(2026, 8, 15, tzinfo=UTC))[-1] == "2627"


def test_url_conforme():
    assert season_url("2526") == (
        "https://www.football-data.co.uk/mmz4281/2526/F1.csv"
    )


# --------------------------------------------------------------------- #
# Immutabilité (le cœur)
# --------------------------------------------------------------------- #
def test_premier_telechargement_ecrit_fichier_horodate(tmp_path: Path):
    t0 = datetime(2026, 5, 31, 14, 30, 5, tzinfo=UTC)
    res = ingest_season("2526", raw_dir=tmp_path, downloader=lambda u: FAKE_V1, now=t0)
    assert not res.deduplicated
    assert res.path.name == "F1_2526_20260531T143005Z.csv"


def test_contenu_identique_est_deduplique(tmp_path: Path):
    t0 = datetime(2026, 5, 31, 14, 0, 0, tzinfo=UTC)
    t1 = datetime(2026, 5, 31, 18, 0, 0, tzinfo=UTC)
    ingest_season("2526", raw_dir=tmp_path, downloader=lambda u: FAKE_V1, now=t0)
    res2 = ingest_season("2526", raw_dir=tmp_path, downloader=lambda u: FAKE_V1, now=t1)
    assert res2.deduplicated and res2.path is None
    assert len(list(tmp_path.glob("F1_2526_*.csv"))) == 1  # pas de doublon


def test_contenu_different_cree_nouvelle_capture(tmp_path: Path):
    t0 = datetime(2026, 5, 31, 14, 0, 0, tzinfo=UTC)
    t1 = datetime(2026, 5, 31, 18, 0, 0, tzinfo=UTC)
    ingest_season("2526", raw_dir=tmp_path, downloader=lambda u: FAKE_V1, now=t0)
    res2 = ingest_season("2526", raw_dir=tmp_path, downloader=lambda u: FAKE_V2, now=t1)
    assert not res2.deduplicated
    assert len(list(tmp_path.glob("F1_2526_*.csv"))) == 2


def test_ecrasement_refuse_et_fichier_intact(tmp_path: Path):
    existing = tmp_path / _capture_name("2526", "20260601T000000Z")
    existing.write_bytes(b"preuve historique")
    t = datetime(2026, 6, 1, 0, 0, 0, tzinfo=UTC)
    with pytest.raises(RawImmutabilityError):
        ingest_season("2526", raw_dir=tmp_path, downloader=lambda u: b"autre", now=t)
    assert existing.read_bytes() == b"preuve historique"  # jamais altéré


def test_telechargement_vide_rejete(tmp_path: Path):
    t = datetime(2026, 5, 31, 14, 0, 0, tzinfo=UTC)
    with pytest.raises(ValueError):
        ingest_season("2526", raw_dir=tmp_path, downloader=lambda u: b"   ", now=t)


def test_ingest_recent_enchaine_les_saisons(tmp_path: Path):
    t = datetime(2026, 5, 31, 12, 0, 0, tzinfo=UTC)
    res = ingest_recent(3, raw_dir=tmp_path, downloader=lambda u: FAKE_V1, now=t)
    assert len(res) == 3
    assert len(list(tmp_path.glob("*.csv"))) == 3
