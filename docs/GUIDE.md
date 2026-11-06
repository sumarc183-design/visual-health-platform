# Guide d'utilisation

Détail de tout ce que le [README](../README.md) résume : installation, données,
exécution, explicabilité, tests et points d'attention techniques.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt   # dépendances + pytest + ruff
pre-commit install                    # optionnel : lint et format à chaque commit
```

Python 3.10 ou plus. `requirements.txt` borne chaque dépendance à la version
majeure validée (Mac Intel et Colab).

## Poids RETFound

Dépôt HuggingFace protégé (« gated ») : créer un compte, accepter les
conditions sur la page du modèle
([RETFound_mae_natureCFP](https://huggingface.co/YukunZhou/RETFound_mae_natureCFP)),
puis créer un token en lecture.

```bash
huggingface-cli login --token VOTRE_TOKEN
python scripts/download_retfound_checkpoint.py \
    --repo-id YukunZhou/RETFound_mae_natureCFP \
    --out third_party/retfound/checkpoints
```

Le fichier fait ~3,7 Go (il contient aussi le décodeur MAE, ignoré au
chargement). Une erreur 401/403 vient du token ou des conditions non
acceptées. Seules les variantes MAE (ViT-L/16) sont compatibles ; les
variantes DINOv2 (ViT-L/14) ne le sont pas.

**Licence** : RETFound est en **CC BY-NC 4.0** — recherche et usage non
commercial uniquement. Voir [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

## Données

RFMiD et ODIR se téléchargent depuis Kaggle sans compte :

```bash
pip install kagglehub
python -c "import kagglehub; print(kagglehub.dataset_download('andrewmvd/retinal-disease-classification'))"
python -c "import kagglehub; print(kagglehub.dataset_download('andrewmvd/ocular-disease-recognition-odir5k'))"
```

Les chemins affichés ont directement la structure attendue par les loaders.
`python scripts/download_rfmid_odir.py --out data/raw` fait la même chose et
les relie dans `data/raw/{rfmid,odir}/` (lien symbolique, `--copy` pour
copier ; `--rfmid-from`/`--odir-from` pour une archive extraite à la main,
copiée sans toucher la source). `scripts/download_isic.py` prépare la phase 2
(dermatologie).

### RFMiD

CSV à 45 colonnes de pathologies + `Disease_Risk` (label binaire à part). Le
sous-ensemble de 28 classes est dérivé du train (classes avec au moins 5
positifs), jamais recalculé sur la validation ou le test, pour éviter la fuite.

### ODIR-5K

Chaque ligne de `full_df.csv` est l'image d'**un** œil, mais les colonnes
`N/D/G/C/A/H/M/O` sont le diagnostic du **patient** (les deux yeux). Les
recopier sur l'image étiquette faux 20,8 % des images (un œil sain de patient
diabétique devient « D » ; 45 % des « O » sont faux).

Par défaut (`--odir-labels eye`), l'étiquette est reconstruite pour l'œil
photographié depuis ses propres mots-clés diagnostiques. Validation : le OU
des deux yeux redonne exactement le diagnostic patient pour 3 358 / 3 358
patients. `--odir-labels patient` reproduit l'ancien protocole.

Le split est **groupé par patient** (les deux yeux d'un patient restent du
même côté) : 70 % train, 15 % validation, 15 % test. La validation est
identique au split d'origine 85/15, sur lequel les premières décisions ont
été prises (étiquettes par œil, prétraitement) ; le test est donc tiré dans
l'ancien train, jamais évalué. Toutes les décisions se prennent sur la
validation, le test n'est lu qu'une fois à la fin. `--odir-test-fraction 0`
redonne le split d'origine sans test (premiers résultats publiés).
`Testing Images/` n'a pas de label (test caché du challenge d'origine) et
n'est pas utilisé.

## Lancer la baseline

```bash
python run_baseline.py --dataset rfmid --data-root <chemin RFMiD> \
    --checkpoint third_party/retfound/checkpoints/RETFound_mae_natureCFP.pth \
    --epochs 30 --save-head heads/rfmid_head.pt --results-json results/rfmid.json
```

Enchaîne contrôle qualité → embeddings RETFound figés (images 224×224
normalisées ImageNet) → tête linéaire multi-label → AUROC macro et par
classe avec IC à 95 %, F1 macro (seuil 0,5), et gap train − validation.
`--seed` fixe l'entraînement de la tête. `--eval-split test` évalue sur le
test (officiel pour RFMiD, 15 % des patients pour ODIR), à lancer une seule
fois, réglages figés sur la validation. Sans `--data-root` : images
synthétiques ; sans `--checkpoint` : poids aléatoires (plomberie uniquement,
signalé en sortie).

Options utiles :

- `--embeddings-cache <dossier>` : le backbone est figé, l'embedding d'une
  image ne dépend que de l'image, du prétraitement, de la taille et du
  checkpoint. Le cache (un `.npz` par combinaison, indexé par image) évite de
  refaire ~10 min de GPU à chaque run ; si tout est en cache, le backbone
  n'est même pas chargé et le run tourne sur CPU en quelques secondes.
  Chaque cache enregistre une empreinte de sa configuration (contenu du
  checkpoint, prétraitement et sa version `PREPROCESS_VERSION`, taille
  d'image) et refuse d'être réutilisé avec une autre : supprimer le fichier
  pour recalculer. Un ancien cache sans empreinte est adopté une fois, avec
  un avertissement ; si le checkpoint est absent de la machine (tête seule
  sur CPU), son empreinte n'est pas vérifiée.
- `--epochs` (défaut 30) et `--head-batch-size` (défaut 16) : réglages de la
  tête des résultats publiés. `--batch-size` ne règle que l'extraction des
  embeddings.
- `--trust-checkpoint` : les poids sont chargés avec
  `torch.load(weights_only=True)`. Si le fichier contient d'autres objets, le
  chargement complet (pickle, qui peut exécuter du code) n'est accepté que
  pour un fichier de `third_party/retfound/checkpoints/` (téléchargé par
  `scripts/download_retfound_checkpoint.py`) ou avec cette option.
- `--predictions <fichier.npz>` : probabilités image par image du jeu évalué,
  avec les étiquettes et les patients. De quoi calculer IC, comparaisons,
  seuils par classe ou calibration sans relancer le GPU.
- `--bootstrap N` (défaut 1000) : IC à 95 % par bootstrap **sur les
  patients** pour ODIR (les deux yeux d'un patient sont corrélés ; les tirer
  image par image rétrécirait les intervalles) ; **par image** pour RFMiD,
  qui ne fournit pas d'identifiant patient. Classes à moins de 10 positifs
  signalées : leur IC est peu fiable.
- `--results-json` : métriques du jeu évalué sous `eval` (`val` est gardé
  pour compatibilité, même contenu, y compris quand `--eval-split test`), et
  un `manifest` : commit du code, empreinte du checkpoint, version du
  prétraitement, réglages de la tête, tailles des jeux, versions de Python,
  torch et NumPy.
- Contrôle qualité : sonde sur 50 images, sans rien retirer. Sur données
  réelles, il lit le fichier d'origine : résolution avant redimensionnement,
  flou sur la rétine recadrée à 512 px, sans le bord du disque. Seuil de flou
  calibré (`DEFAULT_BLUR_THRESHOLD` = 4,16, `scripts/calibrate_quality.py`,
  `results/quality_calibration.json`) : 2e percentile des images
  d'entraînement ODIR acceptables. Il détecte le flou franc, pas les autres
  défauts de qualité (exposition, artefacts).

### Comparer deux variantes

Sur la validation ODIR, une AUROC par classe a un IC de ±0,03 à ±0,09 :
deux runs ne se comparent pas en regardant leurs chiffres séparés.
`scripts/compare_predictions.py` calcule la différence **appariée** (mêmes
images, mêmes tirages de patients pour les deux runs), bien plus précise :

```bash
python scripts/compare_predictions.py results/pred_odir_none_val.npz \
    results/pred_odir_crop-mask_val.npz --json results/compare_odir_val.json
```

Un écart n'est retenu que si son IC exclut 0. Toujours sur la validation.

### Google Colab (GPU gratuit)

[![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sumarc183-design/visual-health-platform/blob/main/notebooks/colab_baseline_retfound.ipynb)

`GPU T4` dans le type d'exécution, token HuggingFace dans les Secrets Colab
sous le nom `HF_TOKEN`, puis `Tout exécuter`. Le notebook récupère les poids,
télécharge RFMiD ou ODIR, évalue les variantes de prétraitement sur la
validation (cache d'embeddings sur Google Drive) et les compare, produit les
planches Grad-CAM et d'occlusion, puis, si `RUN_TEST = True`, le chiffre
final sur le test.

### Prétraitement commun (`crop-mask`)

Avec `--preprocess none` (comportement d'origine), chaque image est redimensionnée
directement en 224×224 : une photo rectangulaire devient un ovale plus ou
moins aplati selon l'appareil. Or ODIR mélange 87 formats d'image et
certains sont très liés à une maladie (60 % des images 3456×3456 sont des
glaucomes, contre 4,9 % en moyenne) : le modèle peut reconnaître l'appareil
plutôt que la maladie, ce que le test d'occlusion a montré sur la DMLA.

`--preprocess crop-mask` (`src/data/fundus_preprocess.py`, **défaut**) rend la géométrie
identique pour toutes les images :

1. recadrage sur le champ rétinien (seuils sur les marges noires et les
   bandes blanches de remplissage, plus long bloc continu de rétine) ;
2. carré par ajout de noir, sans déformer ;
3. masque commun : disque (97 % du demi-côté, efface liseré et encoche)
   coupé en haut et en bas à 89 % du demi-côté. 19 % des images ODIR sont
   tronquées en haut et en bas par l'appareil ; à 0,89, 90 % d'entre elles
   ont exactement la même forme, pour environ 5 % de rétine perdue sur les
   images complètes.

Le prétraitement est enregistré dans la tête (`--save-head`) : Grad-CAM et
l'occlusion rechargent les images de la même façon.

## Validation externe (APTOS 2019)

Pour savoir si la détection de la rétinopathie diabétique tient sur une
base jamais vue (autres patients, appareils et annotateurs que RFMiD et
ODIR) :

1. **GPU** : `notebooks/kaggle_external_aptos.ipynb` (titre Kaggle :
   « Visual Health — embeddings APTOS (validation externe) »), avec la
   compétition `aptos2019-blindness-detection` en input (règlement à
   accepter une fois). Il calcule les embeddings `crop-mask` des 3 662
   images annotées (`scripts/extract_embeddings.py`, même extraction que
   `run_baseline.py`) ; ~10 min sur T4.
2. **CPU, en local** : placer `aptos_crop-mask_224_RETFound_mae_natureCFP.npz`
   et `aptos_labels.csv` dans `embeddings/`, puis

```bash
python scripts/external_validation.py --rfmid-root data/raw/rfmid --odir-root data/raw/odir \
    --embeddings embeddings --out results/external_aptos.json
```

**EyePACS** (seconde base externe, où les mêmes appareils photographient
malades et sains) : `notebooks/colab_external_eyepacs.ipynb` (titre Colab :
« Visual Health — embeddings EyePACS (validation externe) »), 2 500
patients tirés au hasard (~5 000 images), puis
`external_validation.py --external eyepacs --out results/external_eyepacs.json`
(bootstrap et plis groupés par patient).

Têtes « rétinopathie diabétique » entraînées sur RFMiD, sur ODIR et sur les
deux ; seuils choisis sur la validation de la base d'origine ; sur la base externe,
AUROC (toute rétinopathie, stade ≥ 1, et rétinopathie à adresser, stade
≥ 2), sensibilité et spécificité aux seuils hérités, part détectée par
stade. Rien n'est réglé sur APTOS.

## Fine-tuning partiel de RETFound

`scripts/finetune_retfound.py` entraîne les 4 derniers blocs du ViT-L et la
tête sur RFMiD (configuration fixée à l'avance, sans recherche
d'hyperparamètres : AdamW, lr 2e-5 pour les blocs et 1e-3 pour la tête,
augmentations de couleur et de géométrie avec le masque commun réappliqué,
epoch retenue sur la validation). Il lit une fois le test RFMiD et prédit
sur APTOS (`--aptos-root`) en relevant le format d'image.

1. **GPU** : `notebooks/colab_finetune_retfound.ipynb` (titre Colab :
   « Visual Health — fine-tuning partiel RETFound (RFMiD + APTOS) »),
   ~45-60 min sur T4, secrets `HF_TOKEN` et `KAGGLE_API_TOKEN`.
2. **CPU, en local** : placer le contenu du zip dans `results/finetune/`,
   puis

```bash
python scripts/finetune_analysis.py --finetune-dir results/finetune --rfmid-root data/raw/rfmid \
    --embeddings embeddings --out results/finetune_analysis.json
```

Différence appariée avec la tête linéaire figée (validation, test) ; sur
APTOS, AUROC globale et à format égal pour les deux modèles, leurs
différences appariées et l'AUROC par format (bootstrap par groupes de
doublons). Les poids (`finetune_rfmid.pt`, plusieurs centaines de Mo) restent
sur Colab : seules les prédictions et `finetune_rfmid.json` sont versionnées.

## RETFound-Green et fine-tuning sur EyePACS

`notebooks/colab_green_finetune_eyepacs.ipynb` (titre Colab : « Visual Health
— RETFound-Green et fine-tuning sur EyePACS »), ~2 h sur T4, secrets
`HF_TOKEN` et `KAGGLE_API_TOKEN`, deux téléchargements. Le test RFMiD n'y est
pas relu. En local :

```bash
# visual_health_green.zip : les .npz dans embeddings/
python scripts/green_analysis.py --rfmid-root data/raw/rfmid --embeddings embeddings \
    --out results/green_analysis.json
# visual_health_finetune_eyepacs.zip : le contenu dans results/finetune_eyepacs/
python scripts/finetune_analysis.py --finetune-dir results/finetune_eyepacs --rfmid-root data/raw/rfmid \
    --embeddings embeddings --out results/finetune_eyepacs_analysis.json
```

## Prédire sur une image nouvelle

```bash
python predict.py image1.jpg image2.png --head heads/odir_crop-mask_head.pt \
    --checkpoint third_party/retfound/checkpoints/RETFound_mae_natureCFP.pth \
    --thresholds results/decision_odir.json --policy f1 --out predictions.json
```

Même chaîne que l'entraînement (`src/pipeline/inference.py`, testé pour
redonner exactement les probabilités de `run_baseline.py`) : contrôle
qualité sur le fichier d'origine, prétraitement et taille d'image lus dans
la tête, RETFound figé, tête, puis décision par maladie.

- `--thresholds` : seuils choisis sur la validation par
  `scripts/decision_metrics.py`. `--policy f1` : compromis entre maladies
  ratées et fausses alertes ; `--policy sensitivity` : dépistage
  (sensibilité ≥ 0,90 visée), beaucoup plus de fausses alertes. Sans
  fichier de seuils : probabilités seules. Les seuils publiés valent pour
  les têtes du run publié (crop-mask, 30 epochs, graine 42).
- Maladies en preuve insuffisante (moins de 10 cas) : probabilité sans
  décision.
- `--reject-unusable` : une image qui échoue au contrôle qualité n'est pas
  passée dans le modèle (par défaut, l'échec est signalé et le résultat
  donné).
- Les seuils sont **liés à la tête** : `scripts/decision_metrics.py --head
  <tête> --embeddings embeddings` enregistre une sonde de la tête, après
  avoir vérifié qu'elle redonne les prédictions de validation ;
  `predict.py` refuse ensuite les seuils d'une autre tête, même de mêmes
  classes, et tout seuil hors de [0, 1]. Un ancien fichier sans sonde
  n'est accepté qu'avec `--allow-unverified-thresholds`, signalé comme non
  vérifié.
- Refus si la tête a été entraînée avec un autre checkpoint ou une autre
  version du prétraitement (têtes récentes ; les anciennes sont acceptées
  avec un avertissement).

Outil de recherche, non validé cliniquement : les décisions ne sont pas un
diagnostic.

## Explicabilité (Grad-CAM)

```bash
python scripts/gradcam_examples.py --dataset rfmid --data-root <chemin RFMiD> \
    --checkpoint third_party/retfound/checkpoints/RETFound_mae_natureCFP.pth \
    --head heads/rfmid_head.pt --out results/gradcam_rfmid.png
```

Pour les classes les mieux reconnues (ou `--classes`), prend les vrais
positifs de validation les plus sûrs et les montre à côté de leur carte
Grad-CAM. À relire : la chaleur doit tomber sur des structures plausibles
(lésions, disque optique, macula), pas sur un artefact (bords, fond noir,
texte).

Deux pièges propres aux ViT, gérés dans `src/explainability/gradcam.py` :
- un bloc de transformer sort une séquence de tokens, pas une carte 2D :
  `reshape_transform` retire le token CLS et replie les 196 patches en grille
  14×14 ;
- le backbone est gelé : sans entrée dérivable, aucune activation ne porte de
  gradient et Grad-CAM échoue. L'entrée est donc rendue dérivable.

### Contrôle par occlusion

Sur RETFound, Grad-CAM s'allume surtout sur le fond noir et les bords
(planches de `results/`) : tokens « artefacts » de forte norme des ViT MAE,
ou vrai raccourci (forme du masque, caméra) ? L'occlusion tranche sans passer
par les gradients :

```bash
python scripts/occlusion_examples.py --dataset rfmid --data-root <chemin RFMiD> \
    --checkpoint third_party/retfound/checkpoints/RETFound_mae_natureCFP.pth \
    --head heads/rfmid_head.pt --out results/occlusion_rfmid.png --json results/occlusion_rfmid.json
```

Mêmes classes et mêmes images que la planche Grad-CAM. Un carré gris de
32 px glisse sur l'image (pas de 16 px) ; la carte montre la chute du
**logit** de la classe quand chaque zone est cachée (le logit, car à p = 1,00
la probabilité est saturée). Gris = moyenne ImageNet, pas noir : du noir sur
le fond noir ne changerait rien et masquerait un raccourci sur la forme du
masque. Deux mesures par image :

- `fond gris` (`fond Δ` sur les anciennes planches) : variation du logit quand tout le hors-champ (marges noires ou
  blanches) est remplacé par du gris. Proche de 0 : le fond ne compte pas.
  Fortement négatif : raccourci.
- `hors champ a/b` : part de l'effet d'occlusion hors rétine (a) contre part
  de l'image hors rétine (b). a nettement au-dessus de b : le modèle
  s'appuie sur le fond plus que sa surface ne l'explique.

Ces planches portent sur 3 images par classe choisies parmi les mieux
prédites. Pour un contrôle sur la population :
`scripts/occlusion_population.py` tire des malades et des sains au hasard
(graine fixe), passe les mêmes images dans plusieurs têtes et donne la
différence appariée ; `notebooks/kaggle_gpu_checks.ipynb` le lance sur ODIR
(`none` contre `crop-mask`, ~45 min sur T4).

Si l'occlusion tombe sur les lésions alors que Grad-CAM tombe sur le fond,
c'est Grad-CAM qui se trompe sur ce modèle, pas le modèle.

## Tests

```bash
pytest tests/ -v                 # unitaires + bout-en-bout (quelques minutes sur CPU)
pytest tests/ -m "not slow"      # unitaires seulement
ruff check . && ruff format --check .
```

- **Synthétique** (`scripts/generate_synthetic_fundus_dataset.py`) : bruit +
  vignettage, pas de vraie physiologie ; fait tourner contrôle qualité →
  RETFound → classification → Grad-CAM. Utilisé par la CI, qui n'a pas les
  ~10 Go de données réelles.
- **Données réelles** (`tests/test_real_datasets.py`, si `data/raw/{rfmid,odir}/`
  existe) : pipeline sur 16 images RFMiD + 16 ODIR, script Grad-CAM sur la
  validation RFMiD, concordance des étiquettes ODIR par œil avec les
  diagnostics patient. Poids aléatoires (le checkpoint n'est pas versionné) :
  plomberie, pas performance.

## Points techniques

Le vrai checkpoint `RETFound_mae_natureCFP` se charge avec 2 clés absentes
(`fc_norm`, créée à neuf comme dans le fine-tuning officiel) et 106 clés du
décodeur MAE ignorées. La CI vérifie avec un checkpoint synthétique : forward
pass (303 M paramètres, conforme au ViT-L/16), clés `model`/`teacher`,
interpolation des position embeddings (224 → 256), tête multi-label, Grad-CAM.

`configs/config.yaml` est un fichier de référence, non lu par les scripts :
les runs réels utilisent les arguments de `run_baseline.py`.
