# Visual Health Platform

Socle modulaire d'analyse d'images médicales. Premier cas d'usage : classification
et localisation sur images de fond d'œil, backbone [RETFound](https://github.com/rmaphoh/RETFound_MAE)
(modèle fondamental rétinien préentraîné, poids publics).

## Résultats

| Date | Données | Classes | Split | AUROC macro |
|---|---|---|---|---|
| 2026-11-02 | RFMiD | 28 | train officiel (1 920) / val officielle (640) | **0,855** |
| 2026-11-02 | ODIR-5K | 8 (N/D/G/C/A/H/M/O) | par patient, 85/15 (5 439 / 953 images) | **0,806** |

Backbone RETFound (`RETFound_mae_natureCFP`) figé, tête linéaire multi-label
(30 epochs), images 224×224 normalisées ImageNet. Lancé sur Colab GPU T4 via
`notebooks/colab_baseline_retfound.ipynb`, sorties brutes (AUROC par classe)
dans `results/`.

Lecture :
- Sonde sur backbone figé, pas le fine-tuning complet des chiffres publiés
  RETFound : attendre quelques points de moins.
- RFMiD : les classes très rares en validation (TV, ST, RP...) ont une AUROC
  instable, peu interprétable.
- ODIR : les labels N..O sont posés au niveau patient et dupliqués sur les
  deux yeux (voir `src/data/odir_dataset.py`), ce bruit de labels plafonne
  surtout N, D et O. Piste : labels par œil depuis les mots-clés diagnostiques.

## Pourquoi le fond d'œil en premier

Double usage : le pipeline construit ici (contrôle qualité, prétraitement,
classification, localisation) sert directement la piste différenciante
« dépistage TSA depuis la rétine » sans reconstruire le socle deux fois.
RFMiD et ODIR sont publics et immédiatement exploitables ; ISIC (dermatologie)
reste une piste parallèle pour une seconde modalité.

## Principe : socle commun + têtes spécialisées

Pas un modèle unique censé tout reconnaître. Un socle modulaire :

```
1. Contrôle qualité (flou, cadrage, doublons)
2. Identification du type d'image / de la modalité
3. Prétraitement adapté à la modalité
4. Classification (tête spécialisée par indication)
5. Localisation / segmentation des zones suspectes
6. Cartes d'explicabilité (Grad-CAM)
7. Score de confiance
```

Chaque nouvelle indication ajoute une tête sur le socle existant, sans le
reconstruire.

## Données phase 1

| Base | Usage |
|---|---|
| RFMiD | Classification multi-pathologies (fond d'œil) |
| ODIR | Classification multi-pathologies (fond d'œil), volume complémentaire |
| ISIC | Dermatologie — seconde modalité (phase 2) |

Voir `docs/USE_CASES.md` pour l'arbitrage complet entre les trois premiers
cas d'usage envisagés.

## Structure

```
configs/                configuration
scripts/                téléchargement des données, setup RETFound
src/pipeline/           contrôle qualité, identification de modalité
src/models/              backbone RETFound, têtes de classification
src/explainability/     Grad-CAM
src/eval/                métriques (AUROC, etc.)
docs/                    roadmap et arbitrage des cas d'usage
```

## Démarrage

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash scripts/setup_retfound.sh   # clone le code officiel RETFound (voir THIRD_PARTY_NOTICES.md)

# Poids RETFound (dépôt HuggingFace gated -- nécessite un compte + token,
# et d'accepter les conditions sur la page du modèle choisi) :
huggingface-cli login --token VOTRE_TOKEN
python scripts/download_retfound_checkpoint.py \
    --repo-id YukunZhou/RETFound_mae_natureCFP \
    --out third_party/retfound/checkpoints

python scripts/download_rfmid_odir.py --out data/raw
```

huggingface.co est hors de la liste d'hôtes autorisés de l'environnement de
développement utilisé pour ce dépôt (confirmé : la commande ci-dessus renvoie
un 403 de politique réseau, pas une erreur d'identifiants) — à lancer depuis
votre poste ou une instance cloud avec accès internet normal.

## Lancer sur GPU gratuit (Google Colab)

[![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sumarc183-design/visual-health-platform/blob/main/notebooks/colab_baseline_retfound.ipynb)

`notebooks/colab_baseline_retfound.ipynb` récupère les poids RETFound
(token HuggingFace dans les Secrets Colab, nommé `HF_TOKEN`), télécharge
RFMiD ou ODIR depuis Kaggle et lance `run_baseline.py` avec le backbone sur
GPU (`--device cuda`, choisi automatiquement si disponible). Sélectionner
`GPU T4` dans le type d'exécution avant de lancer.

Il produit ensuite une planche Grad-CAM (`scripts/gradcam_examples.py`, à
partir de la tête sauvegardée par `run_baseline.py --save-head`) : pour les
classes les mieux reconnues, les vrais positifs de validation les plus sûrs,
côte à côte avec la zone qui a le plus pesé dans la prédiction. À relire
pour vérifier que le modèle regarde des structures plausibles (lésions,
disque optique, macula) et pas un artefact (bords, fond noir, texte).

## Licence RETFound — important

RETFound est en **CC BY-NC 4.0 (NonCommercial)**, pas MIT. Utilisable
librement pour la phase recherche/reproduction, mais **pas dans un produit
commercial** sans licence séparée des auteurs. Voir `THIRD_PARTY_NOTICES.md`
pour le détail et les options avant un lancement commercial.

## État du code

`src/models/retfound_backbone.py`, `src/models/pos_embed.py` et
`src/explainability/gradcam.py` sont adaptés du dépôt officiel RETFound
(architecture ViT-Large/16, chargement de checkpoint avec interpolation des
position embeddings). Testés ici avec un checkpoint synthétique : forward
pass (303M paramètres, conforme au ViT-L/16 attendu), chargement de
checkpoint avec clés `model`/`teacher`, interpolation pos_embed pour un
changement de résolution (224→256), tête de classification multi-label, et
Grad-CAM adapté aux ViT (un Grad-CAM standard ne fonctionne pas sur un
transformer sans `reshape_transform` — corrigé). Le contrôle qualité
(`src/pipeline/quality_control.py`) est également testé (image nette,
floue, sous-résolue).

RFMiD et ODIR ont été téléchargés et le pipeline validé de bout en bout sur
les vraies images (`tests/test_real_datasets.py`, à poids aléatoires pour la
CI) : voir "Tester sur les vraies données" ci-dessous. Avec le vrai
checkpoint RETFound (CC BY-NC 4.0), premières AUROC obtenues sur RFMiD et
ODIR (voir "Résultats" en haut de ce fichier).

## Tester sans données réelles (smoke test)

`scripts/generate_synthetic_fundus_dataset.py` génère des images de fond
d'œil synthétiques (bruit + vignettage, pas de vraie physiologie) au format
RFMiD/ODIR, pour faire tourner tout le pipeline en attendant le
téléchargement réel :

```bash
pip install -r requirements-dev.txt
pytest tests/ -v                 # tests unitaires + bout-en-bout (~2 min)
pytest tests/ -m "not slow"      # unitaires seulement, rapide
```

`tests/test_pipeline_smoke.py::test_full_pipeline_end_to_end` fait tourner
génération -> contrôle qualité -> RETFound -> classification -> Grad-CAM en
une passe. Gardé pour la CI, qui n'a pas les données réelles (~10 Go).

## Tester sur les vraies données

Une fois RFMiD/ODIR téléchargés et organisés (`scripts/download_rfmid_odir.py`,
voir "Démarrage") dans `data/raw/{rfmid,odir}/` :

```bash
pytest tests/test_real_datasets.py -v
```

`src/data/rfmid_dataset.py` et `src/data/odir_dataset.py` lisent la structure
Kaggle réelle (pas le format `C0..C27` générique du générateur synthétique) :

- **RFMiD** : CSV à 45 colonnes de pathologies + `Disease_Risk` (label binaire
  à part) ; le sous-ensemble de 28 classes du config est dérivé du train set
  (>= 5 exemples positifs), jamais recalculé sur val/test pour éviter la fuite.
- **ODIR** : `full_df.csv` a deux étiquettes différentes par ligne — les
  colonnes `N/D/G/C/A/H/M/O` (diagnostic patient, les deux yeux) et une
  colonne `target`/`labels` par œil (parsing de mots-clés). Le loader utilise
  `N..O`, conforme au config ; `Testing Images/` n'a pas de label (jeu de
  test caché du challenge Kaggle d'origine) donc le split train/val est fait
  par tirage aléatoire fixe sur `Training Images/`.

`tests/test_real_datasets.py` fait tourner tout le pipeline sur ces vraies
images (16 échantillons RFMiD + 16 ODIR) — backbone toujours à poids
aléatoires faute de checkpoint RETFound réel, donc validation de plomberie
uniquement, pas de prédictions interprétables. Sauté automatiquement si
`data/raw/` est absent.

## CI

`.github/workflows/ci.yml` lance la suite de tests à chaque push/PR.

## Roadmap

Voir `docs/ROADMAP.md`.
