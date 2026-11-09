# Attributions et licences

Visual Health Platform : © 2026 Marc Su, sous licence CC BY-NC 4.0 ([LICENSE](LICENSE)).
Les fichiers `src/models/retfound_backbone.py` et `src/models/pos_embed.py` sont
adaptés de RETFound. Les poids des modèles et les jeux de données ne sont pas
distribués avec ce dépôt et restent soumis à leurs propres conditions.

## RETFound (`src/models/retfound_backbone.py`, `src/models/pos_embed.py`)

- Dépôt : https://github.com/rmaphoh/RETFound_MAE
- **Licence : CC BY-NC 4.0 (Attribution - Pas d'utilisation commerciale) — PAS MIT/Apache.**

Le code et les poids préentraînés de RETFound sont libres d'usage **pour la
recherche et un usage non commercial uniquement** (exactement la phase 1 :
reproduction et évaluation interne). Ils **ne peuvent pas** être utilisés dans
un produit ou un service commercial sans licence séparée des auteurs.

**Action nécessaire avant tout lancement commercial de Visual Health** : soit
(a) négocier une licence commerciale avec les auteurs de RETFound, soit (b)
prévoir un backbone sans clause NC — par exemple un préentraînement
auto-supervisé sur nos propres images de fond d'œil, ou un autre modèle
fondamental rétinien publié sous licence permissive (MIT/Apache-2.0). Ne pas
livrer les poids RETFound dans un produit payant tant que ce point n'est pas
réglé.

Le code adapté ici se limite à la construction du modèle et au chargement du
checkpoint (architecture + `interpolate_pos_embed`), fidèle à l'original pour
que les checkpoints officiels se chargent sans renommage.

## RETFound-Green (`src/models/retfound_green.py`)

- Engelmann & Bernabeu, « Training a high-performance retinal foundation
  model with half-the-data and 400 times less compute » (arXiv:2405.00117).
- Dépôt : https://github.com/justinengelmann/RETFound_Green
- **Licence : Justin's Custom Non-Commercial Research Licence (CNCRL).**
  Recherche non commerciale ; elle exclut les entités à but lucratif et les
  projets impliquant une entreprise, sauf accord écrit préalable de l'auteur.
  Les œuvres dérivées distribuées doivent rester sous une licence non
  commerciale reconnue et créditer le projet.
- Les poids ne sont pas versionnés ici.

## DINOv2 (`src/models/timm_backbones.py`)

- Oquab et al., « DINOv2: Learning Robust Visual Features without Supervision ».
  Dépôt : https://github.com/facebookresearch/dinov2, poids chargés via timm
  (`vit_base_patch14_dinov2.lvd142m` et `vit_large_patch14_dinov2.lvd142m`).
- **Licence : Apache 2.0.** Poids non versionnés ici.

## PanDerm (`src/models/timm_backbones.py`)

- Yan et al., « A multimodal vision foundation model for clinical dermatology »,
  Nature Medicine 2025 (arXiv:2410.15038). Dépôt : https://github.com/SiyuanYan1/PanDerm
- **Licence : CC BY-NC-ND 4.0**, « non-commercial academic research purposes » avec
  attribution. Aucun code ni poids de PanDerm n'est copié ici : le modèle est construit
  avec timm (`beit_large_patch16_224`) et les poids PanDerm-Large sont téléchargés à
  l'exécution depuis le Google Drive des auteurs, sans modification.

## Jeux de données

- **RFMiD** et **ODIR-5K** : téléchargés depuis Kaggle (miroirs publics des
  challenges RIADD et ODIR-2019). Respecter les conditions d'usage propres à
  chaque jeu ; les images ne sont pas versionnées dans ce dépôt.
- **ISIC 2019** (BCN_20000, HAM10000, MSK) : CC BY-NC 4.0, à citer selon
  https://challenge.isic-archive.com/data/ ; images non versionnées.
- **IDRiD** (Porwal, Pachade et Kokare) : CC BY 4.0, segmentation utilisée
  pour mesurer la localisation ; images non versionnées.
- **APTOS 2019 Blindness Detection** (Asia Pacific Tele-Ophthalmology
  Society, Aravind Eye Hospital) : données de compétition Kaggle, utilisées
  seulement comme jeu de validation externe, dans un notebook Kaggle, après
  acceptation du règlement de la compétition. Ni les images ni les
  embeddings ni les étiquettes ne sont versionnés ; seules des métriques
  agrégées le sont (`results/external_aptos.json`).
- **EyePACS** (compétition Kaggle « Diabetic Retinopathy Detection », 2015,
  California Healthcare Foundation / EyePACS) : même usage, seconde
  validation externe ; échantillon de patients extrait dans un notebook
  Colab après acceptation du règlement. Rien n'est versionné hors des
  métriques agrégées.
