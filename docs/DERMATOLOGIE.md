# Seconde indication : dermatologie (J60)

Note de choix du 8 novembre 2026, avant tout run. Ce n'est pas un avis juridique : les
licences sont recopiées des pages officielles (voir aussi
[DROITS_DONNEES.md](DROITS_DONNEES.md)).

**Proposition** : même protocole que pour le fond d'œil (modèle figé, tête linéaire,
différences appariées), en comparant un modèle spécialisé en dermatologie, **PanDerm**, à
un modèle généraliste sous licence permissive, **DINOv2**, sur **ISIC 2019** puis **ISIC
2024 SLICE-3D Permissive**. À valider avec le tuteur.

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

## État du code (9 novembre)

Prêts et testés, sans run : chargeur ISIC 2019 (`src/data/isic_dataset.py`, groupes par
lésion, validation tirée en groupes, test officiel sans « UNK » : 21 712 / 3 619 / 6 191
images, aucune lésion partagée entre entraînement et validation), modèle DINOv2
(`src/models/timm_backbones.py`, vérifié avec les vrais poids), extraction
(`extract_embeddings.py --dataset isic2019 --backbone dinov2 --preprocess none`). PanDerm
et le notebook GPU attendent la validation du choix.

## Points à vérifier avant le run

1. Que le préentraînement de PanDerm n'inclut pas les images d'évaluation (sinon la
   mesure serait optimiste, comme EyePACS pour RETFound).
2. Les tailles d'entrée et la normalisation exactes de chaque modèle.
3. Pour ISIC 2019, l'absence d'identifiant patient : regrouper au moins par lésion
   (identifiant fourni) pour le bootstrap et les découpages.
