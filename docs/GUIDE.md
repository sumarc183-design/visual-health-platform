# Guide d'utilisation

Détail de tout ce que le [README](../README.md) résume : installation, données,
exécution, explicabilité, tests et points d'attention techniques.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt   # dépendances + pytest + ruff
pre-commit install                    # optionnel : lint et format à chaque commit
```

Python 3.10 ou plus. `requirements.txt` borne chaque dépendance à la version
majeure validée (Mac Intel et Colab).

## Poids RETFound

Dépôt HuggingFace protégé (« gated ») : créer un compte, accepter les
conditions sur la page du modèle
([RETFound_mae_natureCFP](https://huggingface.co/YukunZhou/RETFound_mae_natureCFP)),
puis créer un token en lecture.

```bash
huggingface-cli login --token VOTRE_TOKEN
python scripts/download_retfound_checkpoint.py \
    --repo-id YukunZhou/RETFound_mae_natureCFP \
    --out third_party/retfound/checkpoints
```

Le fichier fait ~3,7 Go (il contient aussi le décodeur MAE, ignoré au
chargement). Une erreur 401/403 vient du token ou des conditions non
acceptées. Seules les variantes MAE (ViT-L/16) sont compatibles ; les
variantes DINOv2 (ViT-L/14) ne le sont pas.

**Licence** : RETFound est en **CC BY-NC 4.0** — recherche et usage non
commercial uniquement. Voir [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

## Données

RFMiD et ODIR se téléchargent depuis Kaggle sans compte :

```bash
pip install kagglehub
python -c "import kagglehub; print(kagglehub.dataset_download('andrewmvd/retinal-disease-classification'))"
python -c "import kagglehub; print(kagglehub.dataset_download('andrewmvd/ocular-disease-recognition-odir5k'))"
```

Les chemins affichés ont directement la structure attendue par les loaders.
`scripts/download_rfmid_odir.py` sert seulement à ranger un téléchargement
manuel dans `data/raw/{rfmid,odir}/` ; `scripts/download_isic.py` prépare la
phase 2 (dermatologie).

### RFMiD

CSV à 45 colonnes de pathologies + `Disease_Risk` (label binaire à part). Le
sous-ensemble de 28 classes est dérivé du train (classes avec au moins 5
positifs), jamais recalculé sur la validation ou le test, pour éviter la fuite.

### ODIR-5K

Chaque ligne de `full_df.csv` est l'image d'**un** œil, mais les colonnes
`N/D/G/C/A/H/M/O` sont le diagnostic du **patient** (les deux yeux). Les
recopier sur l'image étiquette faux 20,8 % des images (un œil sain de patient
diabétique devient « D » ; 45 % des « O » sont faux).

Par défaut (`--odir-labels eye`), l'étiquette est reconstruite pour l'œil
photographié depuis ses propres mots-clés diagnostiques. Validation : le OU
des deux yeux redonne exactement le diagnostic patient pour 3 358 / 3 358
patients. `--odir-labels patient` reproduit l'ancien protocole.

Le split train/validation est **groupé par patient** (les deux yeux d'un
patient restent du même côté). `Testing Images/` n'a pas de label (test caché
du challenge d'origine) et n'est pas utilisé.

## Lancer la baseline

```bash
python run_baseline.py --dataset rfmid --data-root <chemin RFMiD> \
    --checkpoint third_party/retfound/checkpoints/RETFound_mae_natureCFP.pth \
    --epochs 30 --save-head heads/rfmid_head.pt --results-json results/rfmid.json
```

Enchaîne contrôle qualité → embeddings RETFound figés (images 224×224
normalisées ImageNet) → tête linéaire multi-label → AUROC macro et par
classe, F1 macro (seuil 0,5), et gap train − validation. `--seed` fixe
l'entraînement de la tête. Sans `--data-root` : images synthétiques ; sans
`--checkpoint` : poids aléatoires (plomberie uniquement, signalé en sortie).

### Google Colab (GPU gratuit)

[![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sumarc183-design/visual-health-platform/blob/main/notebooks/colab_baseline_retfound.ipynb)

`GPU T4` dans le type d'exécution, token HuggingFace dans les Secrets Colab
sous le nom `HF_TOKEN`, puis `Tout exécuter`. Le notebook récupère les poids,
télécharge RFMiD ou ODIR, lance la baseline puis la planche Grad-CAM.

## Explicabilité (Grad-CAM)

```bash
python scripts/gradcam_examples.py --dataset rfmid --data-root <chemin RFMiD> \
    --checkpoint third_party/retfound/checkpoints/RETFound_mae_natureCFP.pth \
    --head heads/rfmid_head.pt --out results/gradcam_rfmid.png
```

Pour les classes les mieux reconnues (ou `--classes`), prend les vrais
positifs de validation les plus sûrs et les montre à côté de leur carte
Grad-CAM. À relire : la chaleur doit tomber sur des structures plausibles
(lésions, disque optique, macula), pas sur un artefact (bords, fond noir,
texte).

Deux pièges propres aux ViT, gérés dans `src/explainability/gradcam.py` :
- un bloc de transformer sort une séquence de tokens, pas une carte 2D :
  `reshape_transform` retire le token CLS et replie les 196 patches en grille
  14×14 ;
- le backbone est gelé : sans entrée dérivable, aucune activation ne porte de
  gradient et Grad-CAM échoue. L'entrée est donc rendue dérivable.

## Tests

```bash
pytest tests/ -v                 # unitaires + bout-en-bout (quelques minutes sur CPU)
pytest tests/ -m "not slow"      # unitaires seulement
ruff check . && ruff format --check .
```

- **Synthétique** (`scripts/generate_synthetic_fundus_dataset.py`) : bruit +
  vignettage, pas de vraie physiologie ; fait tourner contrôle qualité →
  RETFound → classification → Grad-CAM. Utilisé par la CI, qui n'a pas les
  ~10 Go de données réelles.
- **Données réelles** (`tests/test_real_datasets.py`, si `data/raw/{rfmid,odir}/`
  existe) : pipeline sur 16 images RFMiD + 16 ODIR, script Grad-CAM sur la
  validation RFMiD, concordance des étiquettes ODIR par œil avec les
  diagnostics patient. Poids aléatoires (le checkpoint n'est pas versionné) :
  plomberie, pas performance.

## Points techniques

Le vrai checkpoint `RETFound_mae_natureCFP` se charge avec 2 clés absentes
(`fc_norm`, créée à neuf comme dans le fine-tuning officiel) et 106 clés du
décodeur MAE ignorées. La CI vérifie avec un checkpoint synthétique : forward
pass (303 M paramètres, conforme au ViT-L/16), clés `model`/`teacher`,
interpolation des position embeddings (224 → 256), tête multi-label, Grad-CAM.

`configs/config.yaml` est un fichier de référence, non lu par les scripts :
les runs réels utilisent les arguments de `run_baseline.py`.
