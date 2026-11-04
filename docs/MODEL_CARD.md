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
avec IC à 95 % par bootstrap (sur les patients pour ODIR, sur les images
pour RFMiD), F1 macro au seuil 0,5 (peu
parlant pour les classes rares, en attendant des seuils par classe), gap
train − validation. Choix des variantes sur la validation par différence
appariée ; le test n'est lu qu'une fois. Test final (`crop-mask`) : AUROC
macro 0,846 sur RFMiD, 0,848 sur ODIR. Résultats détaillés :
[README](../README.md#résultats).

## Limites connues

- **Classes rares** : plusieurs pathologies RFMiD n'ont que quelques
  positifs en validation ou en test ; leur AUROC est instable (RP : 1,0 avec
  2 positifs en test, preuve insuffisante).
- **AUROC ≠ décisions** : F1 macro au seuil 0,5 de 0,26 (RFMiD) et 0,48
  (ODIR). Seuils par classe, sensibilité/spécificité et calibration pas
  encore évalués : le modèle ne fournit pas de décision de dépistage.
- **Statut du test ODIR** : hold-out constitué après les premières
  explorations, disjoint du train actuel par patient, mais tiré du train des
  runs antérieurs. Pas de validation externe.
- **Sonde sur backbone figé** : en dessous des chiffres publiés par RETFound,
  obtenus après fine-tuning complet.
- **Population et appareils** : jeux de données publics de quelques centres ;
  aucune évaluation sur d'autres rétinographes, populations ou qualités
  d'image.
- **Contrôle qualité** : seuil de flou pas encore calibré sur des images
  annotées utilisables/inutilisables ; il signale sur un échantillon et ne
  retire rien, ce n'est pas encore une barrière.
- **Grad-CAM est trompeur sur ce modèle** : il s'allume sur le fond noir
  (artefacts des ViT MAE). L'explication de référence est l'occlusion
  (`scripts/occlusion_examples.py`), qui ne montre elle aussi que *où*, pas
  *pourquoi*.
- **Raccourci d'acquisition (réduit, pas exclu)** : sans prétraitement
  commun, 92 % de l'effet d'occlusion de la DMLA d'ODIR tombait hors de la
  rétine (format d'image lié à l'appareil). Avec `crop-mask`, par défaut, la
  part tombe à 5 % sur les exemples examinés (3 par classe, choisis parmi
  les mieux prédits) et l'AUROC s'améliore. Cela ne prouve pas l'absence de
  biais d'appareil dans la population : des différences de couleur ou de
  netteté peuvent subsister dans la rétine. Le glaucome d'ODIR s'appuie
  encore en partie sur les bords (32 % de l'effet).
- **Écart train/test de la tête** sur RFMiD : AUROC 0,98 contre 0,85. Les
  réglages de tête essayés (balayage archivé, `results/sweep_head_*.json`)
  n'apportent pas de gain démontré.

## Considérations éthiques

Jeux de données publics dé-identifiés. Avant tout produit : licence
RETFound, validation clinique, et évaluation des performances par
sous-groupe (âge, sexe, origine, appareil).
