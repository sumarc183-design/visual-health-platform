# Seconde indication : dermatologie (J60)

Note de choix du 8 novembre 2026, avant tout run. Ce n'est pas un avis juridique : les
licences sont recopiées des pages officielles (voir aussi
[DROITS_DONNEES.md](DROITS_DONNEES.md)).

**Proposition** : même protocole que pour le fond d'œil (modèle figé, tête linéaire,
différences appariées), en comparant un modèle spécialisé en dermatologie, **PanDerm**, à
un modèle généraliste sous licence permissive, **DINOv2**, sur **ISIC 2019** puis **ISIC
2024 SLICE-3D Permissive**. Validé par Marc le 9 novembre (avec DINOv2 ViT-L en plus, voir le plan) ; à présenter au tuteur.

## Modèles candidats

RETFound et RETFound-Green sont préentraînés sur la rétine : ils ne conviennent pas.

| Modèle | Préentraînement | Architecture, entrée, embedding | Poids | Licence | Usage en entreprise |
|---|---|---|---|---|---|
| [PanDerm](https://github.com/SiyuanYan1/PanDerm) | Plus de 2 millions d'images de peau, 4 modalités (dermoscopie, clinique, photo du corps entier, histologie) | ViT-L/16 (et ViT-B/16) ; taille non indiquée | Google Drive, accès non vérifié | CC BY-NC-ND 4.0 : « non-commercial academic research purposes » | Exclu |
| [Derm Foundation](https://huggingface.co/google/derm-foundation) (Google) | Paires image-texte du web, puis télédermatologie (États-Unis, Colombie) et cancer cutané (Australie) | BiT-M ResNet101x3, 448 px, 6 144 dimensions | Hugging Face, après acceptation des conditions | [Health AI Developer Foundations](https://developers.google.com/health-ai-developer-foundations/terms) | Pas exclu par le texte, mais restrictions sur les dispositifs médicaux et les usages interdits ; statut d'une tête entraînée sur ses embeddings à faire vérifier |
| [MONET](https://github.com/suinleelab/MONET) | 105 550 images de dermatologie et textes de la littérature médicale | CLIP ViT-L/14, 224 px | Téléchargement direct, non vérifié | CC BY-NC-SA 4.0 (dépôt) | Exclu |
| [DINOv2](https://github.com/facebookresearch/dinov2) | 142 millions d'images généralistes, aucune de dermatologie déclarée | ViT-S/B/L/g en patches de 14 px | Téléchargement direct | Apache 2.0 | Possible |

Aucune de ces pages ne publie de résultat sur ISIC 2019 ou SLICE-3D : il faudra mesurer.

## Pourquoi cette proposition

- **PanDerm** est le plus proche de la tâche (dermoscopie et photos du corps entier dans
  son préentraînement), mais il est non commercial et sans dérivés : il sert de référence
  de recherche, pas de base de produit.
- **DINOv2** est le seul candidat clairement utilisable en entreprise. La comparaison
  montre ce que coûte un modèle généraliste par rapport au spécialiste.
- **Derm Foundation** est une alternative à ajouter si ses conditions conviennent au
  tuteur ; son entrée de 448 px et ses 6 144 dimensions alourdissent le run.
- **ISIC 2019** (25 331 images, 8 diagnostics, CC BY-NC) permet de comparer aux travaux
  publiés ; **SLICE-3D Permissive** (CC BY, identifiants patients fournis) est le seul
  grand jeu compatible avec un usage commercial, mais ce sont des recadrages de photos du
  corps entier, pas de la dermoscopie, et la tâche est binaire.

## Plan retenu (9 novembre, avant tout run)

Validé par Marc le 9 novembre. Script : `scripts/dermatology_analysis.py` (plan dans sa
docstring) ; notebook : `notebooks/kaggle_dermatology_isic2019.ipynb`.

- **Trois modèles figés** : PanDerm-Large (ViT-L/16, spécialisé peau), DINOv2 ViT-L/14
  (généraliste, **même taille** que PanDerm) et DINOv2 ViT-B/14. Sans le ViT-L, un écart
  mêlerait spécialisation et taille du modèle.
- **Mêmes images pour tous** : protocole d'évaluation de PanDerm (petit côté à 256 px,
  recadrage central 224 px, normalisation ImageNet ; PanDerm garde l'écart-type 0,228 de
  son code). Les images ISIC ne sont pas carrées : plus de redimensionnement simple.
- **Même tête** que le reste du projet, 8 classes en un-contre-tous ; AUROC macro
  (principale) et exactitude équilibrée (mesure officielle du challenge) ; test officiel
  sans « UNK », lu une seule fois.
- **Comparaison principale** : PanDerm moins DINOv2 ViT-L, différences appariées,
  bootstrap par lésion et quasi-doublons (repérés sur DINOv2 ViT-L, le test n'ayant pas
  d'identifiant de lésion).

## Vérifications faites (9 novembre)

1. **Implémentation de PanDerm** : leur code (`classification/models/modeling_finetune.py`
   et `builder.py`) a été lu. Leur modèle est un ViT de type BEiT/CAE sans biais de position
   relatif, token de classe après la normalisation finale. Construit ici avec timm et chargé
   en mode strict (seules les 366 clés `encoder.*` du checkpoint CAE sont gardées ; leur
   code charge en `strict=False`). Sur une même entrée, écart maximal de 3,8e-6 avec leur
   implémentation. Empreinte du fichier : `23405e92f853c6f7`.
2. **Images** : le chargement donne les mêmes pixels que leur transformation torchvision
   (`Resize(256)`, `CenterCrop(224)`) sur les six images ISIC de 600 × 450 testées. Ce n'est
   pas une équivalence générale : l'audit du 9 novembre trouve des écarts sur une image
   synthétique de 334 × 200 (arrondis différents). Les trois modèles reçoivent la même
   transformation du projet ; l'effet sur les scores n'est pas quantifié.
3. **Préentraînement de PanDerm** (article, Methods) : 2 149 706 images de 11 sources
   (cohortes MYM et HOP, MMT, NSSI, Edu1 et Edu2, sous-ensemble ISIC 2024, TCGA-SKCM...).
   **ISIC 2019, HAM10000 et BCN20000 n'y figurent pas** ; ce sont des jeux d'évaluation
   de l'article. Réserves : plusieurs sources ne sont pas publiques, un recouvrement
   d'images ne peut donc pas être exclu ; et PanDerm a été évalué par ses auteurs sur
   HAM10000, inclus dans ISIC 2019 : la mesure n'est pas aveugle pour lui.
4. **Sources officielles** : images et CSV depuis l'archive ISIC
   (`isic-archive.s3.amazonaws.com/challenges/2019`, 9,8 Go et 3,8 Go), décompressés au
   fil du téléchargement ; poids PanDerm depuis leur Google Drive.

## Repères publiés

L'article de PanDerm ne donne pas de résultat dédié à ISIC 2019 avec une sonde linéaire
(seule une ablation, protocole non précisé). Sur HAM10000, il rapporte une AUROC de 0,988
(sonde linéaire, Extended Data Table 8) : nos chiffres ne sont pas directement comparables
(autre jeu, autre tête).

## Résultats

Analyse locale du 9 novembre 2026, après extraction des trois caches Kaggle.
Résultats complets : [dermatology_isic2019.json](../results/dermatology_isic2019.json).
Le plan ci-dessus a été conservé : une seule exécution de l'analyse sur le test,
sans sélection de réglages après lecture des résultats.

21 712 images d'entraînement, 3 619 de validation et 6 191 du test officiel, après
exclusion des « UNK ». Les intervalles à 95 % utilisent 1 000 tirages bootstrap
par groupes, communs aux modèles pour les différences appariées. En validation,
les lésions et les quasi-doublons forment 1 990 groupes (3 paires quasi identiques).
Sur le test, aucun quasi-doublon n'est détecté au seuil fixé (cosinus ≥ 0,999 sur
DINOv2-L) : les 6 191 groupes correspondent donc aux images. Sans identifiants de
lésion sur ce test, une corrélation résiduelle entre images reste possible.

### Test officiel

Chaque valeur est suivie de son intervalle à 95 %. Les mesures sont sur l'échelle
0–1 ; l'exactitude équilibrée est la moyenne des rappels des huit classes.

| Modèle figé | AUROC macro | Exactitude équilibrée |
|---|---|---|
| PanDerm-Large | 0,918 [0,912 ; 0,923] | 0,501 [0,480 ; 0,522] |
| DINOv2 ViT-L/14 | 0,881 [0,874 ; 0,888] | 0,439 [0,421 ; 0,459] |
| DINOv2 ViT-B/14 | 0,872 [0,864 ; 0,880] | 0,441 [0,422 ; 0,460] |

L'exactitude équilibrée de 0,44 à 0,50 s'explique en partie par la construction
de la tête : chaque classe est prédite séparément, sans pondération compensant
le déséquilibre des classes, puis la classe au score le plus haut est retenue
(argmax). Les classes rares sont donc peu choisies : le dermatofibrome (DF,
91 images du test) est prédit sur 52 à 84 images selon le modèle, et les lésions
vasculaires (VASC, 104 images) sur 41 à 44 images. Cela pénalise leurs rappels et
la moyenne des rappels. L'AUROC macro, qui évalue le classement des scores par
classe sans cette décision par argmax, reste la mesure principale ; cette
explication ne modifie ni la tête ni l'analyse.

| Différence appariée | AUROC macro | Exactitude équilibrée |
|---|---|---|
| **PanDerm − DINOv2-L (principale)** | **+0,037 [+0,030 ; +0,043]** | **+0,061 [+0,040 ; +0,084]** |
| PanDerm − DINOv2-B | +0,046 [+0,040 ; +0,053] | +0,059 [+0,039 ; +0,082] |
| DINOv2-L − DINOv2-B | +0,010 [+0,002 ; +0,017] | −0,002 [−0,021 ; +0,019] |

L'intervalle de la comparaison principale exclut zéro : la spécialisation apporte
un gain mesurable dans ce protocole, à taille de modèle comparable. Le passage de
DINOv2-B à DINOv2-L améliore aussi l'AUROC sur le test, sans gain démontré en
exactitude équilibrée. Ces comparaisons secondaires restent distinctes du critère
principal fixé avant le run.

### Gains par classe et validation

Sur le test, PanDerm dépasse DINOv2-L en AUROC sur les huit classes en estimation
ponctuelle. Les gains les plus élevés concernent SCC, BKL, VASC et AK ; le plus
faible concerne DF, dont l'intervalle inclut zéro : aucun gain n'est démontré
pour cette classe seule.

| Classe | PanDerm − DINOv2-L, AUROC [IC à 95 %] |
|---|---|
| Mélanome (MEL) | +0,030 [+0,021 ; +0,039] |
| Nævus (NV) | +0,016 [+0,011 ; +0,021] |
| Carcinome basocellulaire (BCC) | +0,023 [+0,016 ; +0,029] |
| Kératose actinique (AK) | +0,046 [+0,028 ; +0,063] |
| Kératose bénigne (BKL) | +0,058 [+0,042 ; +0,075] |
| Dermatofibrome (DF) | +0,007 [−0,016 ; +0,026] |
| Lésions vasculaires (VASC) | +0,048 [+0,019 ; +0,081] |
| Carcinome épidermoïde (SCC) | +0,066 [+0,041 ; +0,093] |

La validation va dans le même sens : PanDerm dépasse aussi DINOv2-L en AUROC
macro, avec **+0,038 [+0,021 ; +0,055]**. Cette concordance renforce le constat,
sans remplacer la comparaison principale sur le test officiel.

### Contrôle ajouté après le run : images jamais notées

Trouvé le 9 novembre en examinant le routeur de modalité : le fichier officiel du test
marque 78 des 6 191 images avec `score_weight = 0` et `validation_weight = 0`. Elles ne
comptent dans aucun score du challenge ; au moins l'une d'elles est un dessin, pas une
lésion. Un poids nul ne prouve pas que les 78 soient inutilisables. Recalcul sur les scores
archivés, sans réentraînement :

| Images du test | PanDerm − DINOv2-L, AUROC macro |
|---|---|
| Toutes (6 191, plan fixé avant le run) | +0,037 [+0,030 ; +0,043] |
| Sans les 78 jamais notées (6 113) | +0,037 [+0,030 ; +0,044] |
| Score final officiel seulement (5 920) | +0,038 [+0,031 ; +0,044] |

La conclusion ne change pas. Le résultat principal reste celui du plan.

**Comparabilité avec le classement officiel** : notre mesure porte sur 8 classes sans
pondération, hors « UNK ». Le challenge note 9 catégories (dont une hors cible) et applique
des poids ([évaluation ISIC 2019](https://challenge.isic-archive.com/landing/2019/),
[code de calcul](https://github.com/ImageMarkup/isic-challenge-scoring/blob/master/isic_challenge_scoring/classification.py)).
Nos chiffres ne se comparent donc pas directement au classement officiel.

### Réserves et traçabilité

- Il s'agit d'une sonde linéaire sur des modèles figés, pas d'un ajustement complet
  ni d'une validation clinique. L'écart ne permet pas d'isoler causalement le seul
  contenu du préentraînement : architectures et méthodes de préentraînement diffèrent.
- Un recouvrement avec le préentraînement de PanDerm reste possible, ses sources
  n'étant pas toutes publiques. HAM10000, inclus dans ISIC 2019, a déjà servi à
  l'évaluer : cette mesure n'est pas aveugle pour PanDerm.
- Les intervalles reflètent le rééchantillonnage des groupes d'évaluation, pas la
  variabilité entre plusieurs entraînements de la tête.
- Code d'analyse : `5b2292c`, enregistré `5b2292c-dirty` dans le JSON car seul
  `.gitignore` avait été modifié pour protéger les caches, journaux et scores locaux.
  Python 3.12.7, NumPy 1.26.4, scikit-learn 1.9.1, torch 2.2.2 et timm 1.0.30.
- Empreintes déclarées dans les caches : PanDerm `23405e92f853c6f7`, DINOv2-L
  `0d3a8369d3cb28e6`, DINOv2-B `388c468b37acb34e`. Celle de PanDerm correspond à
  l'empreinte attendue. Les poids DINOv2-B sont présents dans
  `third_party/dinov2/dinov2_vitb14_timm.pth`, avec une empreinte locale
  `ba7954cb2db99604`, différente de celle du fichier du run Kaggle. Les poids ont
  été réenregistrés sur Kaggle : deux sérialisations des mêmes tenseurs ne donnent
  pas nécessairement le même fichier ni la même empreinte. Cette différence ne
  démontre donc pas une différence de poids ; leur équivalence n'a pas été vérifiée
  par comparaison des tenseurs. L'empreinte DINOv2-L n'est pas vérifiée contre un
  fichier de poids local.
- Contrôles : archive ZIP intègre ; mêmes 31 522 clés uniques dans les trois
  caches, embeddings finis ; cohérence des moyennes macro, différences, intervalles
  et dimensions des scores ; 7 tests ciblés ISIC/analyse réussis ; lint et format
  vérifiés sur `src`, `scripts` et `tests`. La suite complète n'a pas été relancée.
  Aucun nouvel entraînement ni nouvelle évaluation du test pour ces contrôles.
- Le fichier local `results/dermatology_isic2019.npz` reste ignoré par Git ; son
  éventuel partage nécessite l'accord de Marc.

## Faisabilité sur SLICE-3D Permissive (11 novembre 2026)

Plan fixé avant le run (`scripts/slice3d_feasibility.py`, commit `46765c9`) : seule grande
base de peau compatible a priori avec un usage commercial (CC BY / CC0), recadrages de
photos du corps entier, cible = malin. Pas d'identifiant patient : découpage par centre.
DINOv2-B figé, régression logistique à classes équilibrées. PanDerm écarté (son
préentraînement inclut une partie d'ISIC 2024). Échantillon : tous les malins, bénins
tirés au hasard. Le JSON porte `1875905` : un commit de documentation (droits d'ODIR) a
eu lieu pendant le run, le code est celui de `46765c9`.

| | Malins / bénins | AUROC [IC 95 %] | AUC partielle, sensibilité ≥ 80 % (0 à 0,2) |
|---|---|---|---|
| Centres d'apprentissage (MSK, Queensland), validation croisée | 255 / 4 000 | 0,836 [0,809 ; 0,862] | 0,089 [0,074 ; 0,108] |
| Centres jamais vus (ACEMID, Athènes) | 39 / 2 000 | 0,758 [0,662 ; 0,842] | 0,057 [0,025 ; 0,108] |

**Lecture** : un modèle généraliste utilisable en entreprise sépare en partie les lésions
malignes sur ces photos (le hasard donne 0,5 et 0,02), mais perd nettement d'un centre à
l'autre (0,836 → 0,758), avec un intervalle large faute de cas malins (39). Sur cette
tâche, la mesure officielle à haute sensibilité reste basse : on est loin d'un outil de
tri. Exploratoire : sonde linéaire, plusieurs images d'un même patient possibles dans un
centre (validation croisée optimiste).

## Tête « peau » du socle (12 novembre 2026)

`heads/isic2019_dinov2b_head.pt` : DINOv2-B figé, même tête et mêmes réglages que
l'analyse du 9 novembre, entraînée sur le découpage d'entraînement d'ISIC 2019
(`scripts/save_skin_head.py`). Elle redonne les scores archivés de validation et de test
à 2×10⁻⁷ près : aucune nouvelle mesure. Utilisable dans `predict.py` (probabilités
seules ; aucun seuil n'a été choisi pour la peau).

**Pourquoi DINOv2-B et pas PanDerm**, meilleur de +0,046 d'AUROC macro : c'est le même
modèle que le routeur de modalité (un seul calcul d'embedding par image), sous licence
Apache 2.0 ; PanDerm (CC BY-NC-ND, 1,4 Go) reste l'option de recherche. Les données
ISIC 2019 étant CC BY-NC, la tête reste non commerciale.

**Poids DINOv2-B locaux et poids Kaggle : équivalents.** Sur les 2 659 images ISIC 2019
présentes dans les deux caches (routeur calculé sur le Mac, dermatologie sur Kaggle),
les embeddings ont un cosinus médian de 1,0 (1er centile 0,9999998), contre 0,58 entre
images différentes. La réserve ci-dessus sur les deux empreintes est levée : ce sont
les mêmes poids, sérialisés différemment.

