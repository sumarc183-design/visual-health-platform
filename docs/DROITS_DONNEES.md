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
- **Non commercial, et limité par un règlement de compétition** : APTOS (recherche
  académique et enseignement), EyePACS (seulement pour la compétition et pendant sa durée,
  d'après le texte). Les deux interdisent de publier ou redistribuer les données.
- **Non vérifié** : ODIR (aucune licence trouvée).

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
| DeepDRiD (photos classiques) | Contrôle qualité (notes d'experts), test du routeur (via RetinaMNIST) | CC BY-SA 4.0 (fichier `LICENSE` du dépôt) | Possible avec attribution ; une œuvre adaptée doit être partagée sous la même licence (portée pour un modèle entraîné à faire vérifier) | [dépôt](https://github.com/deepdrdoc/DeepDRiD), doi:10.1016/j.patter.2022.100512 | — |
| ODIR-5K | Entraînement et test | Aucune trouvée | Inconnu ; données fournies par Shanggong Medical Technology | [site du challenge](https://odir2019.grand-challenge.org/) | Lire la page de téléchargement, ou écrire à ODIR@sgex.cn |
| APTOS 2019 | Validation externe | Règlement de compétition, section 7 | Exclu : usage non commercial seulement | [règlement](https://www.kaggle.com/competitions/aptos2019-blindness-detection/rules) | Voir ci-dessous |
| EyePACS (Kaggle 2015) | Validation externe | Règlement de compétition, section « Data » | Exclu ; usage limité à la compétition et à sa durée | [règlement](https://www.kaggle.com/competitions/diabetic-retinopathy-detection/rules) | Voir ci-dessous |

Les pages IEEE DataPort indiquent à la fois un accès ouvert aux utilisateurs connectés et
un abonnement requis : à vérifier sur la page de téléchargement.

**ODIR, vérification du 9 novembre 2026 :** l'accueil du challenge et les pages
[Download](https://odir2019.grand-challenge.org/Download/) et
[dataset](https://odir2019.grand-challenge.org/dataset/) ont renvoyé une erreur HTTP
403 lors de la consultation. Leurs conditions n'ont donc pas pu être lues ; la
licence reste non vérifiée. À faire : consulter ces pages depuis un navigateur
connecté, ou demander les conditions à ODIR@sgex.cn. Aucun message n'a été envoyé.

**ODIR, nouvelle vérification du 11 novembre 2026 (navigateur) :** l'accueil et la
page [dataset](https://odir2019.grand-challenge.org/dataset/) sont lisibles. Elles
décrivent le jeu (5 000 patients, données collectées par Shanggong Medical Technology,
publication « conforme aux règles éthiques et de confidentialité chinoises ») mais
**n'indiquent aucune licence ni condition d'usage**. La page
[Download](https://odir2019.grand-challenge.org/Download/) renvoie toujours
« Forbidden » : elle n'est accessible qu'aux participants inscrits au challenge
(bouton « Join »). Licence toujours non vérifiée. Pistes : s'inscrire au challenge
pour lire la page de téléchargement et ses conditions, ou écrire à ODIR@sgex.cn.
Aucune inscription ni aucun message n'a été fait.

### Règlements APTOS et EyePACS (lus le 8 novembre 2026)

**APTOS 2019**, section 7 A : « You may access and use the Competition Data for
non-commercial purposes only, including for participating in the Competition and on
Kaggle.com forums, and for academic research and education. » Section 7 B : « You agree not
to transmit, duplicate, publish, redistribute or otherwise provide or make available the
Competition Data to any party not participating in the Competition. »

**EyePACS** (California Healthcare Foundation, 2015) : « Participants must use the Data
solely for the purpose and duration of the Competition [...] and agree not to transmit,
duplicate, publish, redistribute or otherwise provide or make available the Data to any
party not participating in the Competition. » Et : « participation in the Competition shall
not be construed as having or being granted a license [...] in any of the Data. »

Conséquences à trancher :
- EyePACS : le texte ne prévoit aucun usage après la compétition, même de recherche. Notre
  validation externe sort a priori de ce cadre.
- APTOS : recherche académique et enseignement seulement ; un projet d'entreprise n'est
  pas clairement couvert.
- Les deux : les étiquettes et les identifiants d'images font partie des données. Plusieurs
  fichiers de `results/` publiés en contiennent (stade par image).

**MedMNIST+** (routeur de modalité : radiographies, frottis, OCT, échographie, scanner,
RetinaMNIST) : CC BY 4.0 pour la collection ([Zenodo](https://zenodo.org/records/10519652)) ;
chaque jeu d'origine garde ses propres conditions, non relues une à une.

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

## Cohorte de confirmation : candidates (11 novembre 2026)

**Le lot EyePACS prévu pour la confirmation n'est probablement pas utilisable** : son
règlement limite l'usage à la compétition et à sa durée (voir plus haut). Quatre bases
de rétinopathie diabétique jamais utilisées dans le projet peuvent le remplacer. Toutes
sont gradées selon l'échelle internationale (0 à 4) ; aucune n'autorise un usage
commercial sans accord.

| Base | Images | Licence lue | Accès | Remarques |
|---|---|---|---|---|
| [BRSET](https://physionet.org/content/brazilian-ophthalmological/1.0.2/) (Brésil, 2010-2020) | 16 266 (8 524 patients) | PhysioNet Credentialed Health Data License 1.5.0 : recherche seulement | Compte PhysioNet accrédité, formation CITI, accord signé | Population nouvelle ; aussi d'autres pathologies (multi-label) ; identifiants patients |
| [DDR](https://github.com/nkicsl/DDR-dataset) (Chine) | 13 673 (9 598 patients, d'après l'article) | CC BY-NC-SA 4.0 (README) : pas d'usage commercial sans accord écrit, partage à l'identique | Téléchargement libre (Google Drive, Baidu) | Découpage et nombre d'images gradées à vérifier après téléchargement |
| [Messidor-2](https://www.adcis.net/en/third-party/messidor2/) (France) | 1 748 (d'après l'article) | Recherche et enseignement, gratuit ; usage commercial non autorisé sans accord | Formulaire sur le site | Petite ; gradation fournie par des tiers |
| [IDRiD, gradation](https://idrid.grand-challenge.org/Data/) (Inde) | 516 (413 entraînement, 103 test) | CC BY 4.0 | IEEE DataPort, compte gratuit | Seule licence ouverte, mais petite ; les 81 images de segmentation déjà utilisées en font probablement partie (non précisé) : à exclure |

Sources consultées le 11 novembre 2026 (pages ci-dessus). Les tailles de DDR et de
Messidor-2 viennent des articles, pas de fichiers téléchargés.

**Proposition, à valider avec le tuteur** : DDR comme cohorte de confirmation principale
(grande, téléchargement libre, aucune image vue par le projet), BRSET en seconde
confirmation si l'accréditation PhysioNet est obtenue. Les deux licences sont non
commerciales : compatibles avec le statut actuel de démonstration, pas avec un produit.
Rien n'a été téléchargé ni demandé.

## Questions à trancher

1. Le projet est-il un « projet impliquant une entreprise » au sens de la licence de
   RETFound-Green ? Si oui : accord écrit de l'auteur, ou abandon de Green.
2. RETFound étant non commercial, quel modèle de base pour un produit vendu : licence des
   auteurs, autre modèle, ou préentraînement sur des données propres ?
3. APTOS et EyePACS : retirer les étiquettes publiées dans `results/` (et de l'historique
   git ?), et décider si ces bases peuvent rester dans le projet. Licence d'ODIR.
4. Pour la dermatologie : partir de SLICE-3D Permissive (compatible a priori) ou des jeux
   CC BY-NC (plus proches de la dermoscopie classique) ?
