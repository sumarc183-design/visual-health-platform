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
bash scripts/setup_retfound.sh
python scripts/download_rfmid_odir.py --out data/raw
```

## Roadmap

Voir `docs/ROADMAP.md`.
