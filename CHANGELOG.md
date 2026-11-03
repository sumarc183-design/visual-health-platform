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
- Grad-CAM sur les vraies prédictions (`scripts/gradcam_examples.py`),
  sauvegarde de la tête (`--save-head`).
- Notebook Google Colab (poids via `HF_TOKEN`, données via kagglehub).
- Résultats : AUROC macro 0,855 sur RFMiD, 0,806 sur ODIR (étiquettes patient).
- CI (lint ruff puis tests), `pyproject.toml`, `pre-commit`, carte du modèle,
  guide d'utilisation, test du classifieur de modalité.

### Corrigé
- Normalisation ImageNet manquante avant RETFound.
- Plantage sur GPU (tête sur `cuda`, embeddings de validation sur CPU).
- Grad-CAM inopérant sur backbone gelé (entrée non dérivable).

### Retiré
- `scripts/smoke_test.py`, doublon du mode synthétique de `run_baseline.py`.
- Évaluation AUROC dupliquée dans `run_baseline.py`, remplacée par
  `src/eval/metrics.py`.
- Dépendances inutilisées (`albumentations`, `pyyaml`, `tqdm`).
