# Changelog

Format : [Keep a Changelog](https://keepachangelog.com/fr/1.1.0/). Versions
selon [SemVer](https://semver.org/lang/fr/).

## [Non publié]

Corrections issues de l'audit du 4 novembre 2026. Résultats publiés
inchangés : prédictions identiques à 5·10⁻⁷ près (RFMiD et ODIR, validation
et test), étiquettes identiques.

### Corrigé
- Contrôle qualité : la résolution était mesurée après redimensionnement en
  224×224 (jamais trop petite) et le flou en convention BGR alors que les
  images sont en RGB. Il lit maintenant le fichier d'origine et mesure le
  flou sur la rétine à échelle fixe, sans le bord du disque.
- Cache d'embeddings : recherche des images manquantes quadratique en nombre
  d'images ; un checkpoint ou un prétraitement modifié sous le même nom
  réutilisait silencieusement des embeddings périmés (empreinte ajoutée).
- Chargement des poids : le repli sans `weights_only` (pickle) était
  automatique pour n'importe quel fichier. Il essaie d'abord en autorisant
  seulement `argparse.Namespace`, puis n'accepte le chargement complet que
  pour un fichier du dossier officiel ou avec `--trust-checkpoint`.
- Notebook : une erreur de `run_baseline.py` passait inaperçue (`| tee`
  renvoie le code de `tee`) ; chaque étape vérifie maintenant sa sortie.
- ODIR : une cellule de mots-clés vide devenait la classe O ; elle lève
  maintenant une erreur. Casse des mots-clés normalisée ; un nom de fichier
  sans `_left`/`_right` est refusé au lieu d'être pris pour l'œil droit.

### Modifié
- `--epochs` vaut 30 par défaut (réglage des résultats publiés, au lieu de 5).
- `--head-batch-size` (défaut 16) rend explicite la taille des mini-batchs
  de la tête ; `--batch-size` ne règle que l'extraction.
- JSON de résultats : métriques sous `eval` (`val` gardé pour
  compatibilité) et `manifest` (commit, empreinte du checkpoint, version du
  prétraitement, réglages, versions). Mêmes informations dans la tête
  sauvegardée.

### Ajouté
- `scripts/aptos_device_check.py`, `notebooks/colab_aptos_formats.ipynb`
  et `results/aptos_device_check.json` : contrôle à appareil égal sur
  APTOS (format d'image d'origine). Dans APTOS, l'appareil donne presque
  le diagnostic (lien 0,336) ; les têtes entraînées avec des images APTOS
  s'en servent (AUROC 0,98-0,99 globale, 0,934-0,964 à appareil égal) ; la
  tête RFMiD reconnaît la maladie à appareil égal (0,942) mais ses scores
  se décalent selon l'appareil (globale 0,830).
- `scripts/site_adaptation.py` et `results/site_adaptation_aptos.json` :
  adaptation à un nouveau site sur APTOS coupé en deux (site / rapport).
  Sans étiquette (centrage, CORAL) : pas de gain ; seuil recalibré avec 50
  images : sensibilité 0,91 mais spécificité 0,39 ; tête réentraînée avec
  50 images : AUROC 0,981, gain suspect à cause du biais d'acquisition
  d'APTOS (analyse de structure ajoutée à `external_validation.py`).
- `results/external_aptos.json` : validation externe de la rétinopathie
  diabétique sur APTOS (3 662 images). AUROC 0,827 (tête RFMiD, 0,947 sur
  son test), 0,526 (ODIR), 0,738 (multi-source) ; seuils non
  transposables (sensibilité 0,51 au lieu de 0,90) ; diagnostic : 0,995
  avec une tête entraînée dans APTOS.
- Validation externe sur APTOS 2019 (rétinopathie diabétique) :
  `src/data/aptos_dataset.py`, `scripts/extract_embeddings.py` (GPU),
  `scripts/external_validation.py` (CPU, seuils hérités de la validation
  d'origine) et `notebooks/kaggle_external_aptos.ipynb`.
- `scripts/calibrate_quality.py` et `results/quality_calibration.json` :
  seuil de flou calibré à 4,16 (2e percentile des images d'entraînement
  ODIR acceptables), désormais par défaut (`DEFAULT_BLUR_THRESHOLD`) dans
  `run_baseline.py` et `predict.py`. Détecte 79 à 90 % du flou franc simulé,
  signale 1,9 % d'ODIR et 0,1 % de RFMiD ; ne détecte pas les 21 images
  « low image quality » d'ODIR (AUROC 0,51), dont le défaut n'est pas le
  flou ; perte du modèle indépendante de la netteté. `openpyxl` en
  dépendance de développement (annotations d'origine d'ODIR).
- `scripts/robustness_checks.py` et `results/robustness_checks.json` :
  appareil retrouvé à partir des embeddings, AUROC à appareil égal
  (`stratification_gap` dans `src/eval/metrics.py`), transfert entre RFMiD
  et ODIR pour la rétinopathie diabétique, la DMLA et la myopie.
- `results/occlusion_population_odir.json` : occlusion sur échantillon tiré
  au hasard (ODIR, Kaggle T4) ; DMLA hors rétine 77 % → 12 % (apparié −65 %
  [−75 ; −50]). `results/predict_check_odir.json` : `predict.py` avec les
  vrais poids redonne les probabilités de l'entraînement (écart ≤ 10⁻⁶).
  `results/results_odir_{none,crop-mask}.json` régénérés avec manifeste
  (mêmes AUROC).
- `scripts/occlusion_population.py` : occlusion sur un échantillon tiré au
  hasard (malades et sains, erreurs comprises), mêmes images pour plusieurs
  têtes, différence appariée ; `notebooks/kaggle_gpu_checks.ipynb` l'enchaîne
  sur ODIR (`none` contre `crop-mask`) avec le contrôle de `predict.py`
  sur les vrais poids.
- `predict.py` et `src/pipeline/inference.py` : prédiction sur une image
  nouvelle avec la même chaîne que l'entraînement (contrôle qualité,
  prétraitement de la tête, RETFound, tête), décision par maladie avec les
  seuils de la validation, refus d'une tête incompatible (checkpoint ou
  version du prétraitement), option `--reject-unusable`.
- `src/eval/decision.py`, `scripts/decision_metrics.py` et
  `results/decision_{rfmid,odir}.json` : seuils par maladie choisis sur la
  validation (F1 maximal ou sensibilité ≥ 0,90), sensibilité, spécificité,
  VPP, VPN, faux positifs et négatifs au test, précision moyenne, Brier,
  ECE, courbes de fiabilité et température ; maladies à moins de 10
  positifs marquées « preuve insuffisante ».
- `scripts/domain_remedies.py` et `results/domain_remedies.json` : remèdes
  comparés sur la validation (normalisation par base, effacement linéaire
  de l'appareil LEACE, tête multi-source) ; tête multi-source retenue et
  lue une fois sur le test.
- `scripts/sweep_head.py` et `results/sweep_head_{rfmid,odir}.json` :
  archive du balayage des réglages de la tête (train et validation
  seulement).

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
