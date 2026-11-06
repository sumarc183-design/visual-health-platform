# Carte du modèle — Visual Health Platform (fond d'œil)

Format inspiré des *Model Cards* (Mitchell et al., 2019).

## Description

Deux modèles de base au choix, indiqués par la tête passée à `predict.py`. Ils
restent figés et seule une tête linéaire est servie.

| | RETFound (`RETFound_mae_natureCFP`) | RETFound-Green |
|---|---|---|
| Architecture | ViT-Large/16, 303 millions de paramètres | ViT-Small/14 à registres, 22 millions |
| Entrée, embedding | 224 px, 1 024 dimensions | 392 px, 384 dimensions |
| Préentraînement | Auto-supervisé (MAE), inclut Kaggle EyePACS | Auto-supervisé, inclut ODIR |
| Têtes disponibles | RFMiD, ODIR | RFMiD |
| Lecture rapide | Référence du projet ; meilleur sur la DMLA de RFMiD | Meilleur classement externe de la rétinopathie |

- **Ce qui est entraîné** : une tête linéaire multi-label (une sortie par
  pathologie, sigmoïde) sur l'embedding. Les analyses externes réentraînent une
  tête à une seule sortie (rétinopathie) avec les mêmes réglages : ce n'est pas la
  tête multi-label servie.
- **Fine-tuning partiel** : expérience seulement, sur RETFound (4 derniers blocs) ;
  `predict.py` ne le charge pas.
- **Tâche** : classement multi-pathologies sur photo de fond d'œil, une image
  pouvant en porter plusieurs. Aucune variante ne fournit de décision de dépistage
  validée.
- **Explicabilité** : test d'occlusion (zones dont le masquage fait chuter
  la prédiction). Grad-CAM est conservé pour comparaison mais trompeur sur
  RETFound (voir les limites).

## Usage prévu

- Recherche et évaluation interne du socle (contrôle qualité, embeddings,
  têtes par indication, explicabilité).
- **Hors périmètre** : tout usage clinique, diagnostic ou aide à la décision
  médicale. Ce n'est pas un dispositif médical, il n'a été ni validé
  cliniquement ni évalué par un comité.
- **Licences** : RETFound est en CC BY-NC 4.0 ; RETFound-Green a sa propre licence
  de recherche non commerciale (voir
  [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)).

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
parlant pour les classes rares ; seuils par maladie dans
`results/decision_*.json`), gap train − validation. Choix des variantes sur la
validation par différence appariée. « Lu une fois » vaut pour une expérience
donnée : les tests et bases externes ont depuis servi à plusieurs analyses, et ne
sont plus une confirmation indépendante d'une hypothèse formulée ensuite. Test
(`crop-mask`) : AUROC macro 0,846 sur RFMiD, 0,848 sur ODIR, et 0,871 sur RFMiD
avec Green. Résultats détaillés :
[README](../README.md#résultats).

## Limites connues

- **Classes rares** : plusieurs pathologies RFMiD n'ont que quelques
  positifs en validation ou en test ; leur AUROC est instable (RP : 1,0 avec
  2 positifs en test, preuve insuffisante).
- **AUROC ≠ décisions** : au seuil 0,5, la plupart des malades sont ratés
  (sensibilité moyenne 0,37 sur RFMiD, 0,44 sur ODIR). Avec des seuils par
  maladie choisis sur la validation pour une sensibilité de 0,90, le test
  donne 0,82 (RFMiD) et 0,90 (ODIR), mais beaucoup de fausses alertes (VPP
  0,29 et 0,35). Probabilités bien calibrées (ECE ≈ 0,02-0,03). Plusieurs
  maladies restent inutilisables en dépistage (hypertension, drusen,
  « autres », anomalies papillaires) ; 14 classes RFMiD sur 28 ont trop peu
  de cas pour fixer un seuil. Le compromis par maladie reste à valider avec
  un ophtalmologue : le modèle ne fournit pas encore de décision de
  dépistage validée.
- **Statut du test ODIR** : hold-out constitué après les premières
  explorations, disjoint du train actuel par patient, mais tiré du train des
  runs antérieurs. Pas de validation externe.
- **Sonde sur backbone figé** : en dessous des chiffres publiés par RETFound,
  obtenus après fine-tuning complet. Un fine-tuning partiel (4 derniers
  blocs) gagne +0,026 en test RFMiD (0,873) ; sur APTOS, son gain global
  (0,974 contre 0,827) disparaît à format égal (−0,012, non significatif), et
  il ne fait pas mieux sur EyePACS (0,650 contre 0,661). RETFound-Green, non
  servi ici, fait mieux en externe (EyePACS 0,726, APTOS 0,958, à tête égale),
  sans gain démontré sur RFMiD et moins bon sur la DMLA (−0,065). `predict.py`
  propose les deux modèles, avec une tête linéaire ; le fine-tuning n'y est
  pas servi.
- **Population et appareils** : jeux de données publics de quelques centres.
  Le format d'image, indicateur indirect de l'appareil, reste lisible dans les
  embeddings même avec `crop-mask`
  (exactitude équilibrée 1,00 sur RFMiD, 0,91 sur ODIR) ; pas d'usage
  détecté par le classifieur (AUROC à format égal), ce qui ne prouve pas son
  absence, et ce contrôle a
  peu de puissance sur ODIR. Entre bases, la performance baisse nettement
  pour la rétinopathie diabétique (0,947 → 0,820 sur RFMiD, 0,783 → 0,637
  sur ODIR) et pour la DMLA évaluée sur ODIR (0,871 → 0,785) : pas de
  validation externe, voir [README](../README.md#robustesse). Effacer
  l'appareil des embeddings (LEACE) ou les normaliser par base n'y remédie
  pas ; une tête entraînée sur les deux bases fait aussi bien que la tête
  de chaque base, sur les deux (macro −0,002 et +0,008 en test), sauf la
  rétinopathie diabétique de RFMiD (−0,022, significatif).
- **Validation externe (APTOS 2019) : généralisation faible pour la
  rétinopathie diabétique.** AUROC 0,827 avec la tête RFMiD (0,947 sur son
  propre test), 0,526 avec la tête ODIR, 0,738 avec la tête multi-source.
  Les seuils réglés sur la validation ne se transposent pas (sensibilité
  0,51 au lieu de 0,90). Une tête entraînée dans APTOS atteint 0,995, chiffre gonflé par le lien
  entre format et diagnostic : cela ne montre pas à lui seul où le transfert
  échoue. Toute utilisation sur un nouveau site
  demande une évaluation et un recalibrage locaux. Réentraîner la tête
  avec quelques dizaines d'images du site donne 0,98 sur APTOS, mais ce
  chiffre est gonflé : dans APTOS, le format d'image d'origine (indicateur
  indirect de l'acquisition) donne presque le diagnostic, et ces têtes
  s'en servent (AUROC à format égal 0,906-0,951). La tête RFMiD classe
  mieux à format égal (0,919) que globalement (0,821) : résultat compatible
  avec un décalage de scores selon l'acquisition, à confirmer ; il ne
  prouve pas que la reconnaissance externe vaut celle de RFMiD. 6,8 %
  d'APTOS sont des doublons d'images, regroupés dans ces analyses.
- **Validation externe (EyePACS, 2 500 patients) : généralisation très
  faible.** AUROC 0,661 [0,639 ; 0,683] avec la tête RFMiD, 0,684 avec la
  tête ODIR, 0,675 avec la multi-source ; 0,680 à format égal pour la tête
  RFMiD, donc le 0,919 d'APTOS ne se transpose pas. Sensibilité 0,02 au seuil
  « dépistage » de la tête RFMiD (visée 0,90). Une tête entraînée dans
  EyePACS atteint 0,785 (performance de cette sonde linéaire, pas un plafond) :
  la tâche y reste difficile sans changement de base, cause non établie. Le pré-entraînement de RETFound inclut Kaggle
  EyePACS (sans étiquette, d'après l'article de RETFound-Green) : cette base
  n'est pas totalement indépendante du backbone. Ce modèle ne doit pas servir à dépister la
  rétinopathie diabétique, y compris avec les seuils fournis.
- **Localisation (IDRiD, 81 images) : les cartes d'occlusion ne prouvent pas que
  RETFound regarde les lésions** (0,598, +0,016 [−0,004 ; +0,039] par rapport à la
  seule position), et elles recouvrent davantage le disque optique. Green les suit
  mieux (0,657, +0,083 [+0,055 ; +0,110]) et évite le disque. L'accord reste
  modeste : ces cartes ne sont pas une preuve clinique et ne doivent pas guider une
  lecture.
- **Contrôle qualité** : seuil de flou calibré le 5 nov. (4,16 ; détecte
  79 à 90 % des images franchement floutées, signale ~2 % d'ODIR). Il ne
  détecte pas les images « mauvaise qualité » annotées dans ODIR (AUROC
  0,51 sur 21 images) : leur défaut n'est pas le flou. Le modèle n'est pas
  moins bon sur les images les moins nettes de RFMiD et ODIR. Il signale
  et ne retire rien par défaut (`predict.py --reject-unusable` pour
  refuser) : ce n'est pas un contrôle de qualité complet (exposition,
  champ, artefacts non mesurés).
- **Grad-CAM est trompeur sur ce modèle** : il s'allume sur le fond noir
  (artefacts des ViT MAE). L'explication de référence est l'occlusion
  (`scripts/occlusion_examples.py`), qui ne montre elle aussi que *où*, pas
  *pourquoi*.
- **Raccourci d'acquisition (réduit, pas exclu)** : sans prétraitement
  commun, 92 % de l'effet d'occlusion de la DMLA d'ODIR tombait hors de la
  rétine (format d'image lié à l'appareil). Avec `crop-mask`, par défaut, la
  part tombe à 5 % sur les exemples examinés (3 par classe, choisis parmi
  les mieux prédits) et l'AUROC s'améliore. Sur un échantillon tiré au
  hasard (15 malades par classe), la baisse est confirmée : 77 % → 12 %,
  différence appariée −65 % [−75 ; −50] ; sans prétraitement, le glaucome,
  l'hypertension et la myopie s'appuyaient aussi sur le fond. Cela ne prouve pas l'absence de
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
