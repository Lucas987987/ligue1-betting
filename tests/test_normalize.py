"""Tests de normalisation — l'assurance-vie du projet (cf. ARCHITECTURE.md §5).

Le bug le plus dangereux est silencieux : une jointure/normalisation qui rate
ampute les données sans erreur. Ces tests garantissent que :
  - aucun match n'est perdu (rows_in == rows_out),
  - aucun nom ne reste non résolu,
  - les variations d'écriture entre sources (PSG : "Paris Saint-Germain" sur
    FBref vs "Paris SG" ailleurs) se résolvent vers le MÊME canonique,
  - Paris FC et Paris SG ne sont JAMAIS confondus (piège de la sur-fusion),
  - un nom inconnu casse bruyamment.

Lancer depuis la racine du repo :  PYTHONPATH=src python -m pytest tests/
"""

from __future__ import annotations

from pathlib import Path

import pytest

from common.teams import TeamResolver, UnknownTeamError
from consolidation.normalize import normalize_matches

MAPPING = Path(__file__).resolve().parents[1] / "config" / "team_mapping.csv"


@pytest.fixture(scope="module")
def resolver() -> TeamResolver:
    return TeamResolver.from_csv(MAPPING)


# --------------------------------------------------------------------- #
# Résolution unitaire
# --------------------------------------------------------------------- #
def test_psg_variantes_memes_canonique(resolver):
    """Les deux écritures de PSG doivent tomber sur le même canonique."""
    par_fbref = resolver.to_canonical("Paris Saint-Germain", source="fbref")
    par_fd = resolver.to_canonical("Paris SG", source="footballdata")
    assert par_fbref == par_fd == "Paris SG"


def test_paris_fc_jamais_confondu_avec_psg(resolver):
    """Piège de la sur-fusion : Paris FC != Paris SG."""
    assert resolver.to_canonical("Paris FC", source="fbref") == "Paris FC"
    assert resolver.to_canonical("Paris FC", source="footballdata") == "Paris FC"
    assert (
        resolver.to_canonical("Paris FC", source="footballdata")
        != resolver.to_canonical("Paris SG", source="footballdata")
    )


def test_tolerance_espaces_parasites(resolver):
    """Les espaces parasites du copier-coller ne doivent pas casser la résolution."""
    assert resolver.to_canonical("  Rennes  ", source="footballdata") == "Rennes"
    assert resolver.to_canonical("Le  Havre", source="clubelo") == "Le Havre"


def test_nom_inconnu_casse_bruyamment(resolver):
    with pytest.raises(UnknownTeamError):
        resolver.to_canonical("Saint-Étienne", source="footballdata")


def test_source_inconnue_rejetee(resolver):
    with pytest.raises(ValueError):
        resolver.to_canonical("Rennes", source="inexistante")


# --------------------------------------------------------------------- #
# Le test critique : aucun match perdu
# --------------------------------------------------------------------- #
def test_aucun_match_perdu_sur_une_journee(resolver):
    """Une journée complète = 9 matchs (18 équipes). rows_in == rows_out,
    zéro non résolu."""
    journee = [
        {"HomeTeam": "Rennes", "AwayTeam": "Marseille"},
        {"HomeTeam": "Lens", "AwayTeam": "Lyon"},
        {"HomeTeam": "Monaco", "AwayTeam": "Le Havre"},
        {"HomeTeam": "Nice", "AwayTeam": "Toulouse"},
        {"HomeTeam": "Brest", "AwayTeam": "Lille"},
        {"HomeTeam": "Angers", "AwayTeam": "Paris FC"},
        {"HomeTeam": "Auxerre", "AwayTeam": "Lorient"},
        {"HomeTeam": "Metz", "AwayTeam": "Strasbourg"},
        {"HomeTeam": "Nantes", "AwayTeam": "Paris SG"},
    ]
    out, report = normalize_matches(journee, resolver, source="footballdata")
    assert report.ok
    assert report.rows_in == report.rows_out == 9
    assert report.unresolved == []
    # Paris FC et Paris SG tous deux présents et distincts après normalisation
    canon = {r["HomeTeam"] for r in out} | {r["AwayTeam"] for r in out}
    assert "Paris FC" in canon and "Paris SG" in canon


def test_mode_non_strict_collecte_les_erreurs(resolver):
    """En mode non strict, on collecte les non-résolus au lieu de lever —
    pour diagnostiquer un mapping incomplet d'un seul coup."""
    rows = [
        {"HomeTeam": "Rennes", "AwayTeam": "Marseille"},
        {"HomeTeam": "Saint-Étienne", "AwayTeam": "Reims"},  # pas en L1 2025-26
    ]
    out, report = normalize_matches(
        rows, resolver, source="footballdata", strict=False
    )
    assert report.rows_in == report.rows_out == 2  # aucune ligne perdue
    assert not report.ok  # mais des non-résolus signalés
    unresolved_names = {name for _, name, _ in report.unresolved}
    assert "Saint-Étienne" in unresolved_names and "Reims" in unresolved_names


def test_les_18_equipes_ont_un_canonique(resolver):
    """Les 18 équipes de la saison doivent toutes être dans la table."""
    assert len(resolver.canonicals) == 18
