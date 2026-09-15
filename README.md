# Visual Health Platform

Socle modulaire d'analyse d'images médicales. Premier cas d'usage : classification
et localisation sur images de fond d'œil, backbone [RETFound](https://github.com/rmaphoh/RETFound_MAE)
(modèle fondamental rétinien préentraîné, poids publics).

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
bash scripts/setup_retfound.sh   # récupérer le checkpoint RETFound (voir THIRD_PARTY_NOTICES.md)
python scripts/download_rfmid_odir.py --out data/raw
```

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
les vraies images (`tests/test_real_datasets.py`, backbone toujours à poids
aléatoires faute de checkpoint RETFound réel) : voir "Tester sur les
vraies données" ci-dessous. Le checkpoint RETFound lui-même (CC BY-NC 4.0)
reste à récupérer manuellement avant un entraînement réel.

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
