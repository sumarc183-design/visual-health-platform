# Changelog

Format : [Keep a Changelog](https://keepachangelog.com/fr/1.1.0/). Versions
selon [SemVer](https://semver.org/lang/fr/).

## [0.1.0] — 2026-11-04

Premier jalon : socle fond d'œil de bout en bout avec les vrais poids RETFound.

### Ajouté
- `run_baseline.py`, point d'entrée unique : contrôle qualité, embeddings
  RETFound figés sur GPU, tête multi-label, AUROC et F1 macro, gap
  train − validation, graine fixée (`--seed`), export JSON (`--results-json`).
- Loaders RFMiD et ODIR sur la structure Kaggle réelle ; étiquettes ODIR par
  œil (`--odir-labels`), split par patient.
- Script Grad-CAM sur les prédictions de la tête entraînée
  (`scripts/gradcam_examples.py`, planches pas encore produites), sauvegarde
  de la tête (`--save-head`).
- `--eval-split test` : évaluation sur le jeu de test officiel RFMiD.
- Intervalles de confiance à 95 % des AUROC (bootstrap sur les patients),
  nombre de positifs par classe, alerte sous 10 positifs.
- `--predictions` : probabilités image par image ; `scripts/compare_predictions.py`
  compare deux runs par différence d'AUROC appariée.
- Split ODIR 70/15/15 par patient (`--odir-test-fraction`, `--eval-split test`
  pour ODIR) : validation inchangée, test tiré dans l'ancien train.
- Cache d'embeddings (`--embeddings-cache`) : le backbone n'est chargé que
  pour les images absentes du cache.
- Notebook : compare `none` et `crop-mask` en un run, cache sur Google Drive,
  test final protégé par `RUN_TEST`.
- Prétraitement commun `--preprocess crop-mask` : recadrage sur la rétine et
  masque identique pour toutes les images, contre le raccourci d'acquisition
  (format d'image lié à la maladie dans ODIR).
- Test d'occlusion (`scripts/occlusion_examples.py`, cellule Colab 5 bis) :
  contrôle des cartes Grad-CAM sur les mêmes images, avec mesure de
  l'effet du fond.
- Notebook Google Colab (poids via `HF_TOKEN`, données via kagglehub).
- Résultats finaux (`crop-mask`, test lu une seule fois) : AUROC macro
  **0,846** sur le test RFMiD (validation 0,864) et **0,848** sur le test
  ODIR (validation 0,861). Premiers résultats avec `none` : RFMiD 0,858
  (validation) et 0,832 (test) ; ODIR 0,845 avec étiquettes par œil, 0,806
  avec étiquettes patient (split 85/15).
- Notebook utilisable sur Kaggle comme sur Colab (secrets, cache local, zip
  des sorties).
- CI (lint ruff puis tests), `pyproject.toml`, `pre-commit`, carte du modèle,
  guide d'utilisation, test du classifieur de modalité.

### Modifié
- `--preprocess crop-mask` devient le défaut : sur la validation, il fait
  +0,020 d'AUROC sur ODIR (IC [+0,011 ; +0,030]), sans coût sur RFMiD, et
  ramène l'effet d'occlusion de la DMLA d'ODIR sur la rétine (92 % → 5 %
  hors champ). `--preprocess none` reproduit les premiers chiffres.

### Corrigé
- Planche d'occlusion : le « Δ » s'affichait en carré (absent de la police).
- La loss affichée à chaque epoch est la moyenne de l'epoch, plus celle du
  dernier batch.
- Occlusion avec `crop-mask` : champ rétinien = masque commun exact, plus un
  seuillage qui comptait les zones sombres de la rétine comme du fond.
- `scripts/download_rfmid_odir.py` copie ou relie les données (kagglehub) au
  lieu de déplacer les fichiers source.
- Un checkpoint RETFound auquel il manque des poids de l'encodeur était
  chargé en silence (couches aléatoires) : il est maintenant refusé.
- Le contrôle qualité annonçait des images « rejetées » qui restaient dans
  le jeu : il les signale désormais comme telles, sans filtrer.
- Normalisation ImageNet manquante avant RETFound.
- Plantage sur GPU (tête sur `cuda`, embeddings de validation sur CPU).
- Grad-CAM inopérant sur backbone gelé (entrée non dérivable).

### Retiré
- `scripts/smoke_test.py`, doublon du mode synthétique de `run_baseline.py`.
- Évaluation AUROC dupliquée dans `run_baseline.py`, remplacée par
  `src/eval/metrics.py`.
- Dépendances inutilisées (`albumentations`, `pyyaml`, `tqdm`).
