# Carte du modèle — Visual Health Platform (fond d'œil)

Format inspiré des *Model Cards* (Mitchell et al., 2019).

## Description

- **Modèle de base** : RETFound (`RETFound_mae_natureCFP`,
  [rmaphoh/RETFound_MAE](https://github.com/rmaphoh/RETFound_MAE)), ViT-Large/16
  préentraîné de façon auto-supervisée (Masked Autoencoder) sur des photos
  couleur de fond d'œil. **Figé** : aucun de ses poids n'est modifié ici.
- **Ce qui est entraîné** : une tête linéaire multi-label (une sortie par
  pathologie, sigmoïde) sur l'embedding de 1 024 dimensions.
- **Tâche** : dépistage multi-pathologies sur photo de fond d'œil ; une image
  peut porter plusieurs pathologies à la fois.
- **Explicabilité** : test d'occlusion (zones dont le masquage fait chuter
  la prédiction). Grad-CAM est conservé pour comparaison mais trompeur sur
  ce modèle (voir les limites).

## Usage prévu

- Recherche et évaluation interne du socle (contrôle qualité, embeddings,
  têtes par indication, explicabilité).
- **Hors périmètre** : tout usage clinique, diagnostic ou aide à la décision
  médicale. Ce n'est pas un dispositif médical, il n'a été ni validé
  cliniquement ni évalué par un comité.
- **Licence** : RETFound est en CC BY-NC 4.0 — aucun usage commercial sans
  licence des auteurs (voir [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)).

## Données d'évaluation

- **RFMiD** (challenge RIADD) : 28 pathologies retenues (au moins 5 positifs
  dans le train), splits train / validation / test officiels.
- **ODIR-5K** : 8 classes (Normal, Diabète, Glaucome, Cataracte, DMLA,
  Hypertension, Myopie pathologique, Autre), split 70/15/15 **groupé par
  patient** (premiers résultats : 85/15 sans test, validation identique). Étiquettes par œil reconstruites depuis les mots-clés
  diagnostiques (le diagnostic patient recopié sur les deux yeux étiquetait
  faux 20,8 % des images).

## Métriques

AUROC macro et par classe (référence du domaine, indépendante du seuil)
avec IC à 95 % par bootstrap sur les patients, F1 macro au seuil 0,5 (peu
parlant pour les classes rares, en attendant des seuils par classe), gap
train − validation. Choix des variantes sur la validation par différence
appariée ; le test n'est lu qu'une fois. Test final (`crop-mask`) : AUROC
macro 0,846 sur RFMiD, 0,848 sur ODIR. Résultats détaillés :
[README](../README.md#résultats).

## Limites connues

- **Classes rares** : plusieurs pathologies RFMiD n'ont que quelques
  positifs en validation ; leur AUROC est instable.
- **Sonde sur backbone figé** : en dessous des chiffres publiés par RETFound,
  obtenus après fine-tuning complet.
- **Population et appareils** : jeux de données publics de quelques centres ;
  aucune évaluation sur d'autres rétinographes, populations ou qualités
  d'image.
- **Seuil de flou** du contrôle qualité encore à calibrer sur des images
  réelles rejetables.
- **Grad-CAM est trompeur sur ce modèle** : il s'allume sur le fond noir
  (artefacts des ViT MAE). L'explication de référence est l'occlusion
  (`scripts/occlusion_examples.py`), qui ne montre elle aussi que *où*, pas
  *pourquoi*.
- **Raccourci d'acquisition (corrigé)** : sans prétraitement commun, 92 %
  de l'effet d'occlusion de la DMLA d'ODIR tombait hors de la rétine (format
  d'image lié à l'appareil). Le prétraitement `crop-mask`, par défaut, le
  ramène à 5 % et améliore l'AUROC. Le glaucome d'ODIR s'appuie encore en
  partie sur les bords (32 % de l'effet).
- **Surapprentissage de la tête** sur RFMiD : AUROC train 0,98 contre 0,85
  en test.

## Considérations éthiques

Jeux de données publics dé-identifiés. Avant tout produit : licence
RETFound, validation clinique, et évaluation des performances par
sous-groupe (âge, sexe, origine, appareil).
