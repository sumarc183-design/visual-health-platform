# Visual Health Platform

[![CI](https://github.com/sumarc183-design/visual-health-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/sumarc183-design/visual-health-platform/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
[![Ruff](https://img.shields.io/badge/code%20style-ruff-261230)](https://docs.astral.sh/ruff/)
[![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sumarc183-design/visual-health-platform/blob/main/notebooks/colab_baseline_retfound.ipynb)

Socle modulaire d'analyse d'images médicales. Premier cas d'usage : dépistage
multi-pathologies sur images de fond d'œil, avec
[RETFound](https://github.com/rmaphoh/RETFound_MAE) (modèle fondamental
rétinien préentraîné) comme backbone. Chaque nouvelle indication ajoute une
tête spécialisée sur le socle commun, sans le reconstruire.

```mermaid
flowchart LR
    A[Image] --> B[Contrôle qualité<br/>flou, résolution]
    B --> C[Modalité<br/>fond d'œil, dermato]
    C --> D[RETFound figé<br/>ViT-L/16]
    D --> E[Tête multi-label<br/>par indication]
    E --> F[AUROC · F1<br/>gap train − validation]
    E --> G[Grad-CAM<br/>zones décisives]
```

## Résultats

Backbone `RETFound_mae_natureCFP` figé, tête linéaire multi-label (30 epochs,
graine 42), images 224×224, prétraitement `crop-mask` (recadrage sur la
rétine + masque commun), GPU T4 (Colab ou Kaggle). Entre crochets : IC à
95 % par bootstrap, sur les patients pour ODIR, sur les images pour RFMiD
(qui ne fournit pas d'identifiant patient).

| Date | Données | Classes | Split | Validation | **Test** |
|---|---|---|---|---|---|
| 2026-11-04 | RFMiD | 28 (27 en test, PTCR absent) | officiels (1 920 / 640 / 640) | 0,864 [0,840-0,891] | **0,846** [0,813-0,886] |
| 2026-11-04 | ODIR-5K, étiquettes par œil | 8 | par patient 70/15/15 (4 474 / 953 / 965) | 0,861 [0,841-0,880] | **0,848** [0,816-0,873] |

Le test n'a été lu qu'une fois par jeu, après le choix du prétraitement sur
la validation. Ses chiffres sont compatibles avec l'IC de la validation : rien
n'indique un optimisme marqué de la validation, sans que cela le prouve.
Le test ODIR est un hold-out constitué le 3 novembre, **après** les premières
explorations : il est disjoint du train actuel (par patient), mais ses images
faisaient partie du train des runs antérieurs (split 85/15). Une confirmation
indépendante demandera des données externes.

**Choix du prétraitement**, par différence d'AUROC appariée sur les mêmes
images de validation (`scripts/compare_predictions.py`,
[RFMiD](results/compare_rfmid_val.json), [ODIR](results/compare_odir_val.json)) :

| | `none` (redimensionnement direct) | `crop-mask` | Différence [IC 95 %] |
|---|---|---|---|
| RFMiD | 0,858 | 0,864 | +0,006 [−0,005 ; +0,021], non significative |
| ODIR | 0,840 | 0,861 | **+0,020 [+0,011 ; +0,030]** |
| ODIR, DMLA | 0,857 | 0,900 | **+0,043 [+0,001 ; +0,092]** |

`crop-mask` ne coûte rien sur RFMiD et gagne sur toutes les classes d'ODIR
(significativement pour Normal, Diabète, Glaucome et DMLA), tout en
réduisant nettement le raccourci d'acquisition sur les exemples examinés
(voir [Explicabilité](#explicabilité)).

- **Très bien reconnues** (AUROC test > 0,94) : rétinopathie diabétique,
  DMLA, trous maculaires, myopie (RFMiD) ;
  cataracte et myopie (ODIR). Les classes à moins de 10 positifs ont un IC
  trop large pour conclure (signalées dans [`results/`](results/)) : par
  exemple RP, AUROC test 1,0 avec 2 positifs, est une preuve insuffisante,
  pas une performance parfaite.
- **AUROC ≠ décisions** : l'AUROC mesure le classement, indépendamment du
  seuil. Au seuil 0,5, le F1 macro n'est que de 0,26 sur RFMiD (0,48 sur
  ODIR) : les maladies rares dépassent rarement 0,5. Seuils par maladie,
  sensibilité/spécificité et calibration : voir
  [Décisions et calibration](#décisions-et-calibration).
- **Écart train/test de la tête** sur RFMiD (AUROC 0,98 contre 0,85) : un
  balayage d'une cinquantaine de réglages (régularisation, pondération des
  classes, standardisation, durée ;
  [script](scripts/sweep_head.py), [RFMiD](results/sweep_head_rfmid.json),
  [ODIR](results/sweep_head_odir.json)) n'apporte pas de gain démontré sur la
  validation : meilleurs écarts +0,003 (RFMiD) et +0,007 (ODIR), maxima de
  144 mesures, donc biaisés vers le haut. L'origine de la limite (features
  figées, classes très rares) reste une hypothèse.
- Sonde sur backbone figé, pas le fine-tuning complet des chiffres publiés
  par RETFound : attendre quelques points de moins.

<details>
<summary>Historique (prétraitement <code>none</code>, 2 et 3 novembre)</summary>

| Date | Données | Split | AUROC macro |
|---|---|---|---|
| 2026-11-03 | RFMiD, validation | officiel | 0,858 |
| 2026-11-03 | RFMiD, test (27 classes) | officiel | 0,832 |
| 2026-11-02 | ODIR, étiquettes patient | par patient 85/15 | 0,806 |
| 2026-11-03 | ODIR, étiquettes par œil | par patient 85/15 | 0,845 |

ODIR : le diagnostic patient recopié sur les deux yeux étiquetait faux
20,8 % des images ; le loader utilise l'étiquette de l'œil photographié
([explication](docs/GUIDE.md#odir-5k)), d'où 0,806 → 0,845. Le passage au
split 70/15/15 (moins d'images d'entraînement) rend ces chiffres non
comparables aux résultats actuels.
</details>

## Explicabilité

Grad-CAM s'allume surtout sur le fond noir et les bords : il est trompeur
sur ce modèle (tokens « artefacts » des ViT MAE). L'explication de référence
est le **test d'occlusion**, qui cache des zones de l'image et mesure la
chute de la prédiction ([RFMiD](results/occlusion_rfmid_crop-mask.png),
[ODIR](results/occlusion_odir_crop-mask.png)).

Avec le redimensionnement direct ([avant](results/occlusion_odir.png)), la
DMLA d'ODIR prenait un **raccourci** : 92 % de l'effet d'occlusion tombait
hors de la rétine. ODIR mélange 87 formats d'image, dont certains très liés
à une maladie (60 % des 3456×3456 sont des glaucomes) : la forme de l'ovale
trahissait l'appareil. `crop-mask` donne la même géométrie à toutes les
images, et, sur les exemples examinés, l'effet revient sur la rétine :

| Classe | Effet hors rétine, `none` → `crop-mask` | Zones qui comptent (exemples examinés) |
|---|---|---|
| DMLA (A, ODIR) | 92 % → **5 %** | ✅ la région maculaire |
| DMLA (ARMD, RFMiD), RS | 34-39 % → 1-6 % | ✅ macula, lésions visibles |
| Myopie (MYA, M) | 15-37 % → 13-25 % | ✅ atrophie péripapillaire, effet diffus |
| Glaucome (G, ODIR) | 39 % → 32 % | 🟡 surtout le disque optique, en partie les bords |
| Cataracte (C) | — | ✅ le voile de toute l'image (aucune zone isolée ne compte) |

Hors rétine, toutes les images sont désormais noires et de même forme : ce
qui reste d'effet sur les bords ne peut plus trahir l'appareil par la
géométrie.

**Portée de ces mesures** : 3 images par classe, choisies parmi les positifs
les mieux prédits, avec un seul réglage d'occlusion (carré gris de 32 px).
Elles illustrent ; elles ne prouvent ni que le raccourci a disparu sur toute
la population (l'appareil peut aussi laisser des différences de couleur, de
netteté ou d'éclairage dans la rétine), ni que les zones utilisées sont les
lésions. À confirmer sur des cas tirés au hasard, erreurs comprises, avec
plusieurs réglages d'occlusion et la relecture d'un ophtalmologue. Contrôles
sur toute la population : voir [Robustesse](#robustesse).

**Occlusion sur un échantillon tiré au hasard** (ODIR, validation : 15 malades
et 15 sains par classe, erreurs comprises, mêmes images pour les deux têtes ;
[`scripts/occlusion_population.py`](scripts/occlusion_population.py),
[résultats](results/occlusion_population_odir.json)). Médiane sur les malades :

| Classe | Effet hors rétine `none` → `crop-mask` | Différence appariée [IC 95 %] | Fond gris (logit) `none` → `crop-mask` |
|---|---|---|---|
| DMLA (A) | 77 % → **12 %** | **−65 % [−75 ; −50]** | −5,8 → +0,4 |
| Glaucome (G) | 81 % → 48 % | −15 % [−36 ; −13] | −5,8 → −2,5 |
| Hypertension (H) | 61 % → 26 % | −29 % [−48 ; −22] | −6,2 → −2,9 |
| Myopie (M) | 37 % → 12 % | −25 % [−40 ; −16] | −5,3 → +1,6 |
| Toutes classes | 48 % → 20 % | −16 % [−24 ; −13] | −3,5 → −0,2 |

- Sans prétraitement commun, le raccourci par le fond était **réel sur la
  population**, et pas seulement pour la DMLA : glaucome, hypertension et
  myopie aussi.
- Avec `crop-mask`, la DMLA s'appuie sur la rétine (12 % hors champ pour 28 %
  de surface, fond gris sans effet).
- L'effet hors champ qui reste avec `crop-mask` (glaucome, rétinopathie
  diabétique, « autres ») ne peut pas être un raccourci d'appareil : ce fond
  est noir et **identique pour toutes les images**, il ne porte aucune
  information. C'est la réaction du modèle à la perturbation elle-même (un
  carré gris posé sur du noir) : pour ces classes, les cartes d'occlusion
  sont à lire avec prudence. La signature d'appareil qui subsiste dans la
  rétine (contrôle 1 de [Robustesse](#robustesse)) n'est pas mesurée par
  l'occlusion.

Méthode :
[GUIDE](docs/GUIDE.md#contrôle-par-occlusion),
[prétraitement](docs/GUIDE.md#prétraitement-commun-crop-mask).

## Décisions et calibration

L'AUROC mesure le classement ; un outil de dépistage décide oui/non. Seuils
par maladie **choisis sur la validation**, appliqués une fois au test
([`scripts/decision_metrics.py`](scripts/decision_metrics.py),
[RFMiD](results/decision_rfmid.json), [ODIR](results/decision_odir.json)).
Moyennes sur les maladies à au moins 10 positifs en validation et en test
(RFMiD : 14 sur 28, les 14 autres sont en **preuve insuffisante** ; ODIR :
8 sur 8) :

| Test | Seuil | Sensibilité | Spécificité | VPP | F1 |
|---|---|---|---|---|---|
| RFMiD | 0,5 fixe | 0,37 | 0,99 | 0,78 | 0,44 |
| RFMiD | F1 maximal (val) | 0,52 | 0,97 | 0,57 | 0,50 |
| RFMiD | dépistage : sensibilité ≥ 0,90 (val) | 0,82 | 0,76 | 0,29 | 0,37 |
| ODIR | 0,5 fixe | 0,44 | 0,94 | 0,61 | 0,48 |
| ODIR | F1 maximal (val) | 0,62 | 0,87 | 0,50 | 0,53 |
| ODIR | dépistage : sensibilité ≥ 0,90 (val) | 0,90 | 0,60 | 0,35 | 0,44 |

- Le seuil de 0,5 rate la majorité des malades : les seuils par maladie
  sont indispensables.
- En dépistage, la sensibilité visée sur la validation se retrouve sur le
  test d'ODIR (0,90) mais pas tout à fait sur RFMiD (0,82), et le prix est
  élevé en fausses alertes. Maladies utilisables en dépistage (sensibilité
  et spécificité ≥ 0,85 au test) : myopie (ODIR ; RFMiD juste en dessous,
  0,84 et 0,99), trou maculaire, rétinopathie diabétique et DMLA (RFMiD) ;
  la cataracte d'ODIR a une spécificité de 0,98 mais une sensibilité de
  0,70. Inutilisables en l'état : hypertension
  (ODIR, AP 0,08), drusen (DN), « autres » (O), excavation et pâleur
  papillaires (ODC, ODP), avec des centaines de fausses alertes.
- **Calibration** : les probabilités sont déjà bien calibrées (ECE 0,023
  sur RFMiD, 0,027 sur ODIR ; température ajustée sur la validation 1,09 et
  1,04, sans gain) : pas de calibrateur à ajouter.
- Le compromis sensibilité/fausses alertes de chaque maladie est une
  décision médicale : à fixer avec un ophtalmologue, sur la validation.

## Robustesse

Trois contrôles sur CPU, à partir des embeddings et des prédictions
sauvegardés, sans aucun réglage choisi
([`scripts/robustness_checks.py`](scripts/robustness_checks.py),
[résultats](results/robustness_checks.json)). L'appareil photo est repéré
par le format d'image d'origine.

**1. L'appareil reste lisible dans les embeddings, même avec `crop-mask`.**
Une régression logistique retrouve l'appareil à partir des embeddings
(exactitude équilibrée sur la validation) :

| | Appareils | Hasard | `none` | `crop-mask` |
|---|---|---|---|---|
| RFMiD | 3 | 0,33 | 1,00 | 1,00 |
| ODIR | 24 | 0,04 | 0,96 | 0,91 |

`crop-mask` supprime la signature géométrique, pas la signature de
l'appareil : elle passe par la rétine elle-même (couleur, netteté,
éclairage). Le prétraitement ne garantit donc pas, à lui seul, un modèle
indépendant de l'appareil.

**2. Le classifieur ne s'aide pas de façon mesurable de l'appareil.** AUROC
globale contre AUROC à appareil égal (paires malade/sain du même appareil,
`stratification_gap` dans [`metrics.py`](src/eval/metrics.py)). Un écart
positif signifierait qu'une partie du classement vient des différences de
prévalence entre appareils.

| Prédictions | Écart moyen | Classes à écart significatif (> 0) |
|---|---|---|
| ODIR validation, `none` | +0,033 | N (+0,044), O (+0,072) |
| ODIR validation, `crop-mask` | +0,015 | O (+0,057) |
| ODIR test, `crop-mask` | +0,000 | aucune |
| RFMiD test, `crop-mask` | +0,017 | DR (+0,009) et des classes à 3-14 positifs |

Pour la DMLA d'ODIR, l'AUROC à appareil égal passe de 0,831 (`none`) à
0,862 (`crop-mask`) en validation, et vaut 0,851 en test (AUROC globale
0,875, écart non significatif). **Limite** : sur ODIR, les maladies sont
concentrées sur certains appareils et les formats sont nombreux, si bien
que seules 4 à 21 % des paires malade/sain sont comparables à appareil
égal : ce contrôle a peu de puissance, il ne détecterait qu'un effet
marqué.

**3. Hors de leur base, les résultats tiennent pour la myopie, en partie
pour la DMLA, mal pour la rétinopathie diabétique.** Tête entraînée sur une
base, évaluée sur le test de l'autre (autres patients, appareils et
annotateurs), comparée à la même tête entraînée dans la base cible :

| Évalué sur | Maladie | Entraîné sur la même base | Entraîné sur l'autre base |
|---|---|---|---|
| RFMiD test | Rétinopathie diabétique | 0,947 | **0,820** |
| RFMiD test | DMLA | 0,957 | 0,947 |
| RFMiD test | Myopie | 0,986 | 0,977 |
| ODIR test | Rétinopathie diabétique | 0,783 | **0,637** |
| ODIR test | DMLA | 0,871 | **0,785** |
| ODIR test | Myopie | 0,998 | 0,996 |

Les définitions ne sont pas identiques d'une base à l'autre (stades de
rétinopathie, myopie « pathologique » dans ODIR), ce qui explique une
partie des écarts. Les chiffres obtenus dans une même base surestiment donc
la performance attendue ailleurs, surtout pour la rétinopathie diabétique :
une validation externe reste nécessaire avant toute conclusion clinique.

**4. Remèdes testés : seul l'entraînement sur les deux bases aide.**
Comparés sur la validation, maladies communes
([`scripts/domain_remedies.py`](scripts/domain_remedies.py),
[résultats](results/domain_remedies.json)) :

| Remède | Appareil lisible (RFMiD / ODIR) | AUROC macro dans la base | Effet |
|---|---|---|---|
| Aucun | 1,00 / 0,91 | 0,864 / 0,861 | référence |
| Normaliser les embeddings par base | 1,00 / 0,91 | 0,859 / 0,863 (centrage) ; 0,823 / 0,853 (z-score) | transfert pas meilleur, myopie dégradée |
| Effacer l'appareil (LEACE) | **0,33 / 0,04** (hasard) | 0,848 / 0,832 | appareil illisible, mais la tâche se dégrade et le transfert reste mauvais |
| **Une tête entraînée sur les deux bases** | 1,00 / 0,91 | — | aussi bon que la tête de chaque base, sur les deux |

L'effacement montre que, dans ces données, l'information d'appareil est
mêlée à de l'information utile : l'enlever coûte plus qu'il ne rapporte,
et le classifieur ne s'en sert déjà pas de façon mesurable (contrôle 2).
Le remède retenu est la **tête multi-source**. Lu une fois sur le test,
contre la tête de chaque base (différence appariée) :

| Test | Rétinopathie diabétique | DMLA | Myopie | Macro |
|---|---|---|---|---|
| RFMiD | 0,925 (−0,022 [−0,041 ; −0,004]) | 0,971 (+0,014) | 0,988 (+0,001) | −0,002 [−0,012 ; +0,010] |
| ODIR | 0,788 (+0,005) | 0,890 (+0,019) | 0,999 (+0,001) | +0,008 [−0,001 ; +0,018] |

Un seul modèle couvre donc les deux bases sans perte moyenne, au prix d'une
petite baisse significative de la rétinopathie diabétique sur RFMiD, que les
définitions différentes des deux bases expliquent probablement. Cela ne
prouve pas la généralisation à une **troisième** source : voir le contrôle 5.

**5. Validation externe : la rétinopathie diabétique se généralise mal.**
APTOS 2019 (3 662 images, Aravind Eye Hospital : autres patients,
appareils et annotateurs), jamais utilisée pour le développement ; rien
n'est réglé dessus, les seuils viennent de la validation de la base
d'origine ([`scripts/external_validation.py`](scripts/external_validation.py),
[résultats](results/external_aptos.json), voir le
[GUIDE](docs/GUIDE.md#validation-externe-aptos-2019)) :

| Tête entraînée sur | Test de sa base | **APTOS, toute rétinopathie** | APTOS, à adresser (stade ≥ 2) | Sensibilité au seuil « dépistage » (visée 0,90) |
|---|---|---|---|---|
| RFMiD | 0,947 | **0,827** [0,814 ; 0,839] | 0,816 | 0,51 (spécificité 0,96) |
| ODIR | 0,787 | **0,526** [0,507 ; 0,545] | 0,602 | 0,76 (spécificité 0,25) |
| RFMiD + ODIR | — | **0,738** [0,722 ; 0,753] | 0,784 | 0,79 (spécificité 0,42) |

- Les chiffres obtenus dans une base **surestiment nettement** la
  performance globale ailleurs : −0,12 pour la tête RFMiD ; la tête ODIR
  est au niveau du hasard sur APTOS. Pour la tête RFMiD, cette baisse vient
  surtout de l'appareil (contrôle 7) : à appareil égal, elle reste à 0,942.
- **Les seuils ne se transposent pas** : le seuil réglé pour 90 % de
  sensibilité sur la validation RFMiD n'en donne que 51 % sur APTOS. Un
  déploiement demanderait de recalibrer les seuils sur des images du site.
- La tête multi-source (contrôle 4) fait **moins bien** que la tête RFMiD
  seule sur APTOS : ajouter ODIR, dont les étiquettes de rétinopathie se
  transposent mal, dégrade la généralisation. Le choix « multi-source »
  valait pour couvrir RFMiD et ODIR, pas pour une base nouvelle.
- Diagnostic : une tête entraînée **dans** APTOS (validation croisée)
  atteint 0,995 : les embeddings APTOS séparent bien les classes, donc la
  préparation des images n'est pas en cause et l'échec vient du changement
  de base. Mais ce 0,995 est probablement **gonflé** (ci-dessous).
- **APTOS a un biais d'acquisition** : regroupées sans étiquette, ses images
  forment des groupes presque purs (85 % des images dans un groupe à moins
  de 10 % ou plus de 90 % de malades ; lien groupes/diagnostic 0,367),
  contre 31 % et 0,059 pour RFMiD, 0 % et 0,015 pour ODIR. Malades et sains
  semblent photographiés dans des conditions différentes. Une tête
  **entraînée** sur APTOS peut s'en servir ; la tête RFMiD, qui ne l'a
  jamais vue, non : la mesure externe ci-dessus reste valable.

**6. Adaptation à un nouveau site : à confirmer sur une base sans biais
d'acquisition.** APTOS coupé en deux moitiés stratifiées : la moitié
« site » sert à adapter, la moitié « rapport » à mesurer
([`scripts/site_adaptation.py`](scripts/site_adaptation.py),
[résultats](results/site_adaptation_aptos.json)) :

| Adaptation (tête RFMiD) | AUROC (rapport) | Sensibilité | Spécificité |
|---|---|---|---|
| Aucune (seuil de la validation RFMiD) | 0,830 | 0,50 | 0,95 |
| Sans étiquette : centrage par domaine | 0,773 | — | — |
| Sans étiquette : CORAL | 0,833 | — | — |
| Seuil recalibré avec 50 images étiquetées du site | 0,830 | 0,91 | 0,39 |
| Tête réentraînée avec 50 images étiquetées du site | 0,981 | 0,85 | 0,97 |
| Tête réentraînée avec 400 images étiquetées du site | 0,990 | 0,90 | 0,98 |

- Sans étiquette, rien n'aide. Recalibrer le seuil retrouve la sensibilité
  visée, mais avec 60 % de fausses alertes.
- Réentraîner la tête avec quelques dizaines d'images du site fait bondir
  l'AUROC, **mais sur APTOS ce gain est suspect** : la tête peut apprendre
  le biais d'acquisition de la base plutôt que la maladie. On ne peut pas en
  conclure que « 50 images suffisent » ; il faut refaire l'expérience sur
  une base sans ce biais, ou à appareil égal (métadonnées d'appareil
  d'APTOS à récupérer).
- Pistes restantes pour la généralisation : sources d'entraînement plus
  variées, normalisation des couleurs et augmentation de données,
  fine-tuning partiel de RETFound.

**7. À appareil égal : la maladie se reconnaît, c'est l'échelle des scores
qui dépend de l'appareil.** L'appareil d'APTOS est repéré par le format
d'image d'origine ([`notebooks/colab_aptos_formats.ipynb`](notebooks/colab_aptos_formats.ipynb),
[`scripts/aptos_device_check.py`](scripts/aptos_device_check.py),
[résultats](results/aptos_device_check.json)).

- **Dans APTOS, l'appareil donne presque le diagnostic** : sur 17 formats,
  presque tous ont 0-7 % ou 95-100 % de malades (lien format/diagnostic
  0,336). Seules 4 % des paires malade/sain sont comparables à appareil
  égal : les estimations ci-dessous reposent surtout sur le seul format
  mélangé (2588×1958, 57 % de malades).

| Tête (moitié « rapport » d'APTOS) | AUROC globale | AUROC à appareil égal | Écart [IC 95 %] |
|---|---|---|---|
| RFMiD, jamais vu APTOS | 0,830 | **0,942** | −0,112 [−0,135 ; −0,088] |
| RFMiD + 50 images du site | 0,981 | 0,934 | +0,048 [+0,030 ; +0,067] |
| RFMiD + 200 images du site | 0,989 | 0,945 | +0,044 [+0,029 ; +0,061] |
| Entraînée sur la moitié « site » | 0,995 | 0,964 | +0,032 [+0,019 ; +0,047] |

- Les têtes entraînées avec des images APTOS **profitent du lien
  appareil/diagnostic** (écart positif) : leur AUROC globale de 0,98-0,99
  est gonflée ; à appareil égal, elles ne font pas mieux que la tête RFMiD.
- La tête RFMiD **reconnaît la maladie aussi bien qu'à domicile** (0,942 à
  appareil égal, contre 0,947 sur le test RFMiD), mais **ses scores se
  décalent d'un appareil à l'autre**, au point que des yeux sains de
  certains appareils passent devant des malades d'autres appareils. La
  baisse de l'AUROC globale (0,830) vient de ce décalage, pas d'une perte
  de reconnaissance.
- Conséquence : le remède le plus prometteur n'est pas de réentraîner sur
  le site, mais de **recalibrer les scores par appareil ou par site**, ce
  qui demande des images du site, et peu ou pas d'étiquettes. À tester sur
  une base où malades et sains partagent les mêmes appareils.

## Démarrage rapide

**Sans installation** : bouton *Open in Colab* ci-dessus, `GPU T4`, token
HuggingFace dans les Secrets Colab (`HF_TOKEN`), `Tout exécuter`.

**En local** (poids RETFound et données : voir le [guide](docs/GUIDE.md#poids-retfound)) :

```bash
pip install -r requirements-dev.txt
python run_baseline.py --dataset rfmid --data-root <chemin RFMiD> \
    --checkpoint third_party/retfound/checkpoints/RETFound_mae_natureCFP.pth --epochs 30
pytest tests/                   # unitaires + bout-en-bout sur images synthétiques
```

**Prédire sur une image nouvelle** (tête sauvegardée par `--save-head`,
seuils par maladie choisis sur la validation) :

```bash
python predict.py mon_fond_d_oeil.jpg --head heads/odir_crop-mask_head.pt \
    --checkpoint third_party/retfound/checkpoints/RETFound_mae_natureCFP.pth \
    --thresholds results/decision_odir.json --policy f1   # ou sensitivity (dépistage)
```

Contrôle qualité sur le fichier d'origine, même prétraitement que
l'entraînement, probabilité et décision par maladie (aucune décision pour
les maladies en preuve insuffisante). Outil de recherche : pas un
diagnostic.

## État du socle

| Étape | État |
|---|---|
| 1. Contrôle qualité (flou, résolution) | 🟡 Seuil de flou calibré ([résultats](results/quality_calibration.json)) : détecte le flou franc, pas les autres défauts de qualité ; signale sans filtrer par défaut |
| 2. Identification de la modalité | 🟡 Architecture en place, pas encore entraînée |
| 3. Prétraitement fond d'œil | ✅ `crop-mask` (recadrage + masque commun), normalisation ImageNet, 224×224 |
| 4. Classification multi-label | ✅ RFMiD et ODIR, AUROC mesurée |
| 5. Localisation des zones suspectes | ⬜ J60 |
| 6. Explicabilité | 🟡 Grad-CAM peu fiable sur ce ViT (chaleur sur le fond) ; l'occlusion le remplace. Voir [Explicabilité](#explicabilité) |
| 7. Score de confiance calibré | 🟡 Probabilités déjà bien calibrées (ECE ≈ 0,02-0,03) et seuils par maladie ; à valider cliniquement (jalon J60). Voir [Décisions et calibration](#décisions-et-calibration) |
| 8. Prédiction sur image nouvelle | ✅ `predict.py` ; avec les vrais poids, mêmes probabilités que l'entraînement (écart ≤ 10⁻⁶ sur 20 images, [contrôle](results/predict_check_odir.json)) |

## Données

| Base | Usage |
|---|---|
| RFMiD | Fond d'œil, 28 pathologies (phase 1) |
| ODIR-5K | Fond d'œil, 8 classes, volume complémentaire (phase 1) |
| ISIC | Dermatologie, seconde modalité (phase 2) |
| AI-Hub | Dépistage TSA depuis la rétine (phase 2/3, accès à négocier) |

Arbitrage des cas d'usage : [docs/USE_CASES.md](docs/USE_CASES.md).

## Structure

```
run_baseline.py         point d'entrée : contrôle qualité -> RETFound -> tête -> AUROC
src/pipeline/           contrôle qualité, identification de modalité
src/data/               loaders RFMiD et ODIR (structure Kaggle réelle)
src/models/             backbone RETFound, tête de classification
src/explainability/     Grad-CAM adapté aux ViT
src/eval/               métriques multi-label (AUROC, F1)
scripts/                poids RETFound, Grad-CAM, données, générateur synthétique
notebooks/              notebook Google Colab
tests/                  tests unitaires, bout-en-bout synthétique, données réelles
results/                sorties des runs réels
configs/                hyperparamètres de référence (non lus par les scripts)
docs/                   guide, carte du modèle, roadmap, cas d'usage
```

## Documentation

| Document | Contenu |
|---|---|
| [GUIDE.md](docs/GUIDE.md) | Installation, poids, données, exécution, Grad-CAM, tests |
| [MODEL_CARD.md](docs/MODEL_CARD.md) | Usage prévu, données, métriques, limites |
| [ROADMAP.md](docs/ROADMAP.md) | Jalons J30 / J60 / J90 |
| [USE_CASES.md](docs/USE_CASES.md) | Arbitrage entre fond d'œil, dermatologie et TSA |
| [CHANGELOG.md](CHANGELOG.md) | Historique des versions |

## Licences et attributions

⚠️ **RETFound est sous licence CC BY-NC 4.0** : recherche et usage non
commercial uniquement, pas dans un produit commercial sans licence des
auteurs. Détails et options : [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
Pour citer ce travail : [CITATION.cff](CITATION.cff).

Projet de recherche, pas un dispositif médical.
