# Changelog

Format : [Keep a Changelog](https://keepachangelog.com/fr/1.1.0/). Versions
selon [SemVer](https://semver.org/lang/fr/).

## [0.1.0] — 2026-11-03

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
- Test d'occlusion (`scripts/occlusion_examples.py`, cellule Colab 5 bis) :
  contrôle des cartes Grad-CAM sur les mêmes images, avec mesure de
  l'effet du fond.
- Notebook Google Colab (poids via `HF_TOKEN`, données via kagglehub).
- Résultats : AUROC macro 0,855 sur RFMiD (validation), 0,832 sur son test
  officiel ; 0,845 sur ODIR avec étiquettes par œil (0,806 avec étiquettes
  patient).
- CI (lint ruff puis tests), `pyproject.toml`, `pre-commit`, carte du modèle,
  guide d'utilisation, test du classifieur de modalité.

### Corrigé
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
