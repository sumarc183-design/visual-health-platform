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
2. **Images** : le chargement donne exactement les mêmes pixels que leur transformation
   torchvision (`Resize(256)`, `CenterCrop(224)`) sur de vraies images ISIC.
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
