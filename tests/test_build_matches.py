"""Tests de consolidation (build_matches) — garantit qu'aucun match n'est perdu
et que toutes les colonnes football-data sont préservées malgré l'hétérogénéité
entre saisons.

Lancer depuis la racine :  PYTHONPATH=src python -m pytest tests/
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from common.teams import UnknownTeamError
from consolidation.build_matches import build_matches, latest_capture_per_season

MAPPING = Path(__file__).resolve().parents[1] / "config" / "team_mapping.csv"

# Saison 2425 : AVEC Referee, BOM en tête, ligne vide finale.
S2425 = (
    "\ufeffDiv,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR,Referee,"
    "B365H,B365D,B365A,PSH,PSD,PSA\n"
    "F1,16/08/2024,21:00,Rennes,Marseille,2,1,H,M Dupont,2.10,3.40,3.20,2.12,3.45,3.25\n"
    "F1,17/08/2024,17:00,St Etienne,Monaco,0,3,A,J Martin,4.50,3.80,1.75,4.55,3.85,1.78\n"
    ",,,,,,,,,,,,,,\n"
)
# Saison 2526 : SANS Referee, moins de colonnes de cotes.
S2526 = (
    "Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR,B365H,B365D,B365A\n"
    "F1,15/08/2025,20:45,Paris FC,Angers,1,0,H,1.50,4.00,6.00\n"
    "F1,16/08/2025,17:00,Lens,Lyon,2,2,D,2.60,3.30,2.70\n"
)


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    (tmp_path / "F1_2425_20260531T171513Z.csv").write_text(S2425, encoding="utf-8")
    (tmp_path / "F1_2526_20260531T171514Z.csv").write_text(S2526, encoding="utf-8")
    # Capture plus ancienne de la même saison : ne doit PAS être retenue.
    (tmp_path / "F1_2526_20260101T120000Z.csv").write_text(
        S2526.replace("2,2,D", "0,0,D"), encoding="utf-8"
    )
    return tmp_path


def test_prend_la_capture_la_plus_recente(raw_dir: Path):
    caps = latest_capture_per_season(raw_dir)
    assert set(caps) == {"2425", "2526"}
    assert caps["2526"].name == "F1_2526_20260531T171514Z.csv"


def test_aucun_match_perdu_lignes_vides_ignorees(raw_dir: Path, tmp_path: Path):
    rep = build_matches(
        raw_dir=raw_dir, out_path=tmp_path / "matches.csv",
        mapping_path=MAPPING, strict=False, write=True,
    )
    assert rep.total_rows == 4
    assert rep.rows_per_season == {"2425": 2, "2526": 2}


def test_union_colonnes_heterogenes(raw_dir: Path, tmp_path: Path):
    rep = build_matches(
        raw_dir=raw_dir, out_path=tmp_path / "matches.csv",
        mapping_path=MAPPING, strict=False, write=True,
    )
    # Dérivées en tête, Referee (présent seulement en 2425) dans l'union.
    assert rep.columns[:3] == ["season", "HomeTeamCanonical", "AwayTeamCanonical"]
    assert "Referee" in rep.columns
    assert all(c in rep.columns for c in ["B365H", "PSH", "PSD", "PSA"])


def test_equipe_non_mappee_signalee_pas_perdue(raw_dir: Path, tmp_path: Path):
    out = tmp_path / "matches.csv"
    rep = build_matches(
        raw_dir=raw_dir, out_path=out, mapping_path=MAPPING,
        strict=False, write=True,
    )
    assert "St Etienne" in rep.unresolved_teams
    assert not rep.ok
    # La ligne existe quand même, nom brut préservé, canonique vide.
    with out.open(encoding="utf-8") as fh:
        written = list(csv.DictReader(fh))
    assert len(written) == 4
    steti = [r for r in written if r["HomeTeam"] == "St Etienne"][0]
    assert steti["HomeTeamCanonical"] == ""
    assert steti["HomeTeam"] == "St Etienne"  # brut intact


def test_mode_strict_leve(raw_dir: Path, tmp_path: Path):
    with pytest.raises(UnknownTeamError):
        build_matches(
            raw_dir=raw_dir, out_path=tmp_path / "m.csv",
            mapping_path=MAPPING, strict=True, write=False,
        )
