# Droits sur les données et les modèles

Relevé du 8 novembre 2026, demandé pour un projet d'entreprise. Ce n'est pas un avis
juridique : ce document recopie ce que disent les textes lus et liste ce qui reste à
trancher. Une licence peut changer : vérifier à nouveau avant tout usage commercial.

## En bref

- **Usage commercial a priori possible, en citant la source** : RFMiD, IDRiD (CC BY 4.0),
  ISIC SLICE-3D Permissive (CC BY).
- **Non commercial** : RETFound (CC BY-NC 4.0), RETFound-Green (licence propre, qui
  exclut aussi les projets impliquant une entreprise sauf accord écrit), la plupart des
  jeux ISIC (CC BY-NC).
- **Non vérifié** : APTOS et EyePACS (règlements Kaggle illisibles sans connexion), ODIR
  (aucune licence trouvée).

## Modèles

| Modèle | Licence | Usage commercial ou en entreprise | Source |
|---|---|---|---|
| RETFound | CC BY-NC 4.0 | Exclu sans licence des auteurs | [dépôt](https://github.com/rmaphoh/RETFound_MAE), [THIRD_PARTY_NOTICES](../THIRD_PARTY_NOTICES.md) |
| RETFound-Green | Licence propre (CNCRL) | Exclu, y compris pour un projet où une entreprise apporte moyens ou personnel, sauf accord écrit préalable | [LICENSE](https://github.com/justinengelmann/RETFound_Green/blob/main/LICENSE) |

## Jeux de données du fond d'œil

| Jeu | Usage dans le projet | Licence lue | Usage commercial | Source | À faire |
|---|---|---|---|---|---|
| RFMiD | Entraînement et test | CC BY 4.0 | Possible avec attribution (doi:10.21227/s3g7-st65) | [IEEE DataPort](https://ieee-dataport.org/open-access/retinal-fundus-multi-disease-image-dataset-rfmid) | Licence du miroir Kaggle utilisé non lue |
| IDRiD | Localisation | CC BY 4.0 (aussi dans le `LICENSE.txt` téléchargé) | Possible avec attribution (doi:10.21227/H25W98) | [IEEE DataPort](https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid) | — |
| ODIR-5K | Entraînement et test | Aucune trouvée | Inconnu ; données fournies par Shanggong Medical Technology | [site du challenge](https://odir2019.grand-challenge.org/) | Lire la page de téléchargement, ou écrire à ODIR@sgex.cn |
| APTOS 2019 | Validation externe | Règlement Kaggle non lu | Inconnu | [règlement](https://www.kaggle.com/competitions/aptos2019-blindness-detection/rules) | Lire l'onglet « Rules », connecté |
| EyePACS (Kaggle 2015) | Validation externe | Règlement Kaggle non lu | Inconnu | [règlement](https://www.kaggle.com/competitions/diabetic-retinopathy-detection/rules) | Lire l'onglet « Rules », connecté |

Les pages IEEE DataPort indiquent à la fois un accès ouvert aux utilisateurs connectés et
un abonnement requis : à vérifier sur la page de téléchargement.

## Dermatologie (seconde indication, J60)

| Jeu ISIC | Images | Tâche | Licence | Patient identifié |
|---|---|---|---|---|
| 2018, tâche 3 (HAM10000) | 10 015 entraînement | 7 diagnostics | CC BY-NC | Non indiqué |
| 2019 | 25 331 entraînement | 8 diagnostics (9 au test) | CC BY-NC | Non (identifiant de lésion seulement) |
| 2020 (SIIM-ISIC) | 33 126 entraînement | Mélanome contre bénin | CC BY-NC | Oui |
| 2024 SLICE-3D | 401 059 entraînement | Cancer cutané confirmé, binaire | CC BY-NC | Oui |
| **2024 SLICE-3D Permissive** | 217 477 entraînement | Idem | **CC BY** | Oui |

Source : [ISIC Challenge Datasets](https://challenge.isic-archive.com/data/) et
[conditions d'utilisation de l'archive](https://www.isic-archive.com/terms-conditions).
Ces conditions interdisent par ailleurs toute « utilisation commerciale du Site », sans
préciser si cela couvre les jeux de challenge téléchargés : point à faire trancher.

SLICE-3D Permissive est le seul grand jeu ISIC compatible a priori avec un usage
commercial. Ce sont des recadrages de photos du corps entier, pas des images de
dermoscopie : la tâche et la qualité d'image diffèrent des autres jeux ISIC.

## Questions à trancher

1. Le projet est-il un « projet impliquant une entreprise » au sens de la licence de
   RETFound-Green ? Si oui : accord écrit de l'auteur, ou abandon de Green.
2. RETFound étant non commercial, quel modèle de base pour un produit vendu : licence des
   auteurs, autre modèle, ou préentraînement sur des données propres ?
3. Règlements Kaggle d'APTOS et d'EyePACS, et licence d'ODIR.
4. Pour la dermatologie : partir de SLICE-3D Permissive (compatible a priori) ou des jeux
   CC BY-NC (plus proches de la dermoscopie classique) ?
