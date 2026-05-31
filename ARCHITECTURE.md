Architecture — Outil de paris Ligue 1
Document de référence : relevé des décisions de cadrage. Court et décisionnel par choix.
Il acte ce qui a été tranché et pourquoi, pas un cours complet sur chaque sujet.
0. Principe directeur
Projet indépendant du projet tennis (repo, code et pipeline séparés). On transfère les
principes (Signal × Fiabilité, capture des closing lines, discipline CLV, rigueur
anti-overfitting), pas le code.
Discipline de fond, héritée du tennis : valider avant de construire, ne pas miser tant que
le modèle n'est pas validé hors échantillon. Le CLV est le juge de paix.
1. Faisabilité des données (validée)
Source
Contenu
Accès
Cadence
Remarques
football-data.co.uk
Résultats + cotes 1X2 multi-bookmakers (dont Pinnacle)
CSV, code F1
2×/semaine
Source centrale. ~25 saisons d'historique.
ClubElo
Elo par équipe et par date
API CSV gratuite, sans auth
hebdo
api.clubelo.com/{date} ou /{club}. Elo déjà calculé, gère l'avantage domicile.
FBref
xG, xGA, stats avancées
Scraping HTML (pas d'API)
hebdo
Rate limit strict (~1 req/3 s), risque de bannissement IP. À isoler.
The Odds API
Cotes live + closing lines
API (clés gratuites)
week-end
sport_key = soccer_france_ligue_one, marché h2h (3 issues, "Draw" inclus). 5 clés × 500 req/mois = 2500/mois.
Conclusion : faisable, et meilleur point de départ que le tennis (cotes + résultats dans un
même CSV, Elo déjà calculé, xG gratuits). La difficulté n'est pas l'accès aux données mais la
modélisation des 3 issues et la faible taille d'échantillon (~380 matchs/saison).
2. Modèle (décidé)
Dixon-Coles bayésien hiérarchique comme modèle cible. Séquencement :
Dixon-Coles fréquentiste comme baseline qui marche (référence de comparaison).
Version bayésienne ensuite (incertitude quantifiée → alimente la composante Fiabilité).
Cœur mathématique
Buts modélisés par Poisson : log(λ) = intercept + home_adv + att[i] − def[j] (domicile),
log(μ) = intercept + att[j] − def[i] (extérieur).
Correction Dixon-Coles : facteur τ appliqué aux seuls 4 scores faibles (0-0, 1-0, 0-1, 1-1),
contrôlé par ρ. ρ ressort négatif en pratique (~ −0.1 à −0.15) → rehausse 0-0 et 1-1.
Garde-fous : prior resserré sur ρ, vérifier que λμρ ne s'approche pas de 1 (sinon log-vraisemblance
explose → divergences MCMC). Renormaliser la matrice des scores avant de sommer les régions 1/N/2.
Décroissance temporelle : φ(t) = exp(−ξ·Δt). ξ est un hyperparamètre (choisi par
validation, pas estimé dans le modèle).
Priors (régularisation = arme anti-overfitting principale)
Code
Contrainte d'identifiabilité obligatoire : somme(att) = 0 et somme(def) = 0,
sinon l'échantillonneur ne converge pas.
Sortie
Pour chaque match, l'inférence renvoie une distribution de P(1/N/2) (pas un point) :
moyenne = prédiction, dispersion = fiabilité native. C'est ce qui justifie l'effort bayésien.
Note : "le plus fiable" ≠ "le plus confiant". On vise la calibration (quand le modèle dit
60 %, ça arrive ~60 % du temps), pas la confiance affichée. Le ML (XGBoost, réseaux) est écarté
au début : mal calibré et trop confiant sur petit échantillon.
3. Validation (décidée)
Walk-forward (jamais de k-fold mélangée → data leakage). On ne prédit jamais un match avec
une information postérieure à son coup d'envoi.
Amorçage : réserver ~1 à 1,5 saison pour l'entraînement initial avant d'enregistrer des
prédictions évaluables.
Découpage 3 blocs temporels :
Train (saisons anciennes) → ajuste les paramètres.
Validation (saisons du milieu) → choix de ξ et des réglages.
Test (saison la plus récente) → évalué une seule fois, jamais touché avant. Brûlé dès
qu'on le regarde pour décider quoi que ce soit.
Fenêtre extensible (pas glissante) : la décroissance ξ dévalue déjà le vieux passé.
Métriques (par ordre de priorité)
Log-loss — métrique reine, guide le choix de ξ.
Brier multiclasse — complémentaire.
Courbe de calibration — test visuel de la fiabilité.
CLV vs clôture Pinnacle — juge de paix.
Baseline obligatoire : comparer la log-loss au marché (cotes de clôture dévigorisées).
Si le modèle ne bat pas le marché en log-loss, il n'a pas d'edge prédictif — quel que soit le
ROI apparent (probablement de la variance sur 380 matchs).
4. Architecture du pipeline (5 étages découplés)
Communication uniquement par fichiers dans data/. Aucun module n'importe un module d'un
étage voisin : ils ne se connaissent que par les CSV (à schéma fixe) qu'ils lisent/écrivent.
Pipeline rejouable, testable étage par étage, robuste si une source tombe.
Code
Décisions structurantes
Découplage Fit / Predict (clé de voûte) : le fit MCMC est lent (plusieurs min à ~20 min sur
runner gratuit 2 cœurs) → 1×/semaine. La prédiction recharge posterior.nc et calcule en
secondes → quotidienne. On ne réajuste pas tout le modèle à chaque journée.
Crons séparés par source (cadences différentes) ; FBref isolé hors chemin critique :
s'il tombe, le pipeline continue avec les derniers xG connus.
The Odds API / closing lines : une requête /odds ramène tous les matchs d'un coup (pas de
conso par match). Capturer la closing seulement pour les matchs au coup d'envoi imminent (<1-2 h).
Cadencer pour préserver le quota (2500 req/mois).
Point dur signalé : mapping des noms d'équipes
C'est là que ce type de projet échoue silencieusement. "Paris SG" / "Paris Saint-Germain" /
"Paris S-G" / "Paris Saint Germain FC" selon les sources. Une jointure ratée perd des matchs
sans erreur visible → modèle entraîné sur données amputées. Parade : table de correspondance
canonique (config/team_mapping.csv) maintenue à la main, vérifiée dès le début. Base possible :
teamname_replacements.json de soccerdata.
5. Structure des fichiers (décidée)
Code
Décisions de structure
src/ (prod) ≠ validation/ (offline) : frontière physique qui matérialise "le bloc test
ne touche jamais la prod". La validation se lance à la main, jamais en cron.
model/ scindé en 3 : dixoncoles.py = définition unique du modèle, importée par fit.py
ET predict.py → fit et prédiction utilisent rigoureusement le même modèle.
common/ seul transverse : utilitaires neutres (I/O, noms, dates) sans logique métier.
io.py centralise et valide les schémas → garde-fou contre le match perdu.
config/ en 3 fichiers : toucher la science sans casser l'infra.
Contrat d'étage = son fichier de sortie à schéma fixe : on peut réécrire l'intérieur d'un
étage sans toucher aux autres tant que le schéma tient.
Garde-fou de test prioritaire
test_normalize.py est l'assurance-vie du projet : vérifier qu'après consolidation le nombre de
matchs = attendu (38 journées × 10 = 380/saison) et qu'aucune équipe ne reste non-résolue.
6. Frontend repensé pour 3 issues
Affichage 3 colonnes (1/N/2) : proba modèle + intervalle crédible + cote + EV par issue.
Le nul traité comme issue à part entière (souvent sous-joué par le grand public → valeur).
Signal × Fiabilité sur vecteur : l'intervalle crédible bayésien alimente directement la
Fiabilité (intervalle large = fiabilité basse). Avantage que le tennis binaire n'avait pas.
100 % statique : CSV de l'étage 4 lus en JS côté client, hébergé sur Pages.
7. État du cadrage
Faisabilité ✓ · Modèle ✓ · Validation ✓ · Architecture ✓ · Structure ✓
Prochaine brique de code : consolidation/normalize.py + config/team_mapping.csv
(tout le reste en dépend).
