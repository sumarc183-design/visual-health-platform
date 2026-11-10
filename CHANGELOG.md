# Changelog

Format : [Keep a Changelog](https://keepachangelog.com/fr/1.1.0/). Versions
selon [SemVer](https://semver.org/lang/fr/).

## [Non publié]

### Corrigé
- IDRiD (10 nov.) : localisation relancée sur Kaggle pour obtenir le manifeste des
  archives (modèle, empreintes du checkpoint et des têtes, prétraitement, commit
  `93aa9d5`) ; cartes, tokens et logits identiques au bit près aux archives précédentes.
  Point restant de l'audit du 6 nov. ; l'inventaire des images et masques bruts n'est
  pas encore dans le manifeste (audit du 9 nov., constat 8).
- Audit du 9 nov. : identifiants et étiquettes APTOS/EyePACS retirés des fichiers suivis
  (4 archives de scores gardées en local ; tirages few-shot séparés du rapport
  `site_adaptation_aptos.json`) et test qui empêche leur retour, sur le contenu des
  fichiers. L'historique git n'est pas réécrit.
- Archives de prédictions validées avant tout calcul (clés uniques, dimensions,
  probabilités finies dans [0, 1]) : une liaison tête/seuils avec des NaN ou une
  comparaison avec des clés en double sont refusées (`src/eval/predictions.py`).
- Détecteur qualité, test humain : réponses « incertain » rapportées à part au lieu de
  compter comme « aucun défaut », taux par patient (3/18 et 4/18 au lieu de 3/19 et
  4/19) ; clés de cache versionnées pour les défauts simulés. Résultat initial archivé.
- Documentation : statut du routeur synchronisé, contrôle du routeur non indépendant
  signalé, équivalence des pixels ISIC bornée aux images testées, différence avec le
  score officiel ISIC précisée.

### Ajouté
- Score de confiance (J60, 10 nov.) : compromis cas manqués / fausses alertes pour 1 000
  personnes, par maladie, à 80, 90 et 95 % de sensibilité visée (seuils choisis sur la
  validation). `docs/CONFIANCE.md`, `scripts/operating_points.py`, 1 test.
- Routeur, test de résistance (10 nov.) : 800/800 fonds d'œil DeepDRiD reconnus, dont
  438 de qualité insuffisante, aucun vers la dermoscopie (probabilité max 0,001) ;
  l'erreur résiduelle (rétine très pathologique) n'est pas liée à la qualité d'image.
- Contrôle qualité, seuil d'avertissement (analyse secondaire) : à 10 % de fausses
  alertes fixées sur la validation DeepDRiD, 40 % des images insuffisantes repérées à
  l'évaluation (5 % pour le détecteur actuel) ; 21 % de fausses alertes sur ODIR.
- Contrôle qualité (10 nov.) : détecteur appris sur les notes d'experts de DeepDRiD
  (CC BY-SA 4.0, 1 062 images de qualité insuffisante). Évaluation : sensibilité 0,67
  contre 0,05 pour le détecteur actuel, spécificité 0,57 contre 0,97, AUROC 0,69
  [0,62 ; 0,75] ; défauts graves bien repérés. Transfert ODIR : 21/31 patients repérés
  (3/31 actuel) et 10/18 dans la relecture (4/18), avec plus de fausses alertes. Usage
  envisagé : avertissement, pas rejet. `src/data/deepdrid.py`, 2 tests.
- Routeur de modalité, protocole 2 (10 nov.) : groupes disjoints, inconnus médicaux
  (radiographies, frottis) à l'ajustement ; au test, 0/756 images médicales jamais vues
  (OCT, échographie, scanner) envoyées à un modèle spécialisé, 300/300 fonds d'œil
  DeepDRiD reconnus. Même erreur croisée qu'au protocole 1 (une rétine très pathologique
  vers la dermoscopie) : critère toujours non atteint. `docs/MODALITE.md`,
  `results/modality_router_v2.json`.
- Contrôle qualité (10 nov.) : détecteur appris sur de vrais défauts ODIR. Faute
  d'exemples de défauts généraux hors série relue, il apprend la poussière : 17/26
  patients repérés en validation (3/26 pour l'actuel), mais 25/110 fausses alertes ;
  défauts généraux non repérés. AUROC 0,68 [0,58 ; 0,77]. `results/quality_real_detector.json`.
- Contrôle qualité (9 nov.) : relecture humaine de 36 images ODIR en série équilibrée
  (défaut vu sur 16/18 images à mot-clé, 3/18 témoins) ; détecteur appris sur des
  défauts simulés, sans gain sur les vrais défauts (AUROC 0,56 [0,46 ; 0,67]).
  `docs/QUALITE_EXPOSITION.md`, `results/quality_learned_detector.json`. 7 tests.
- Routeur de modalité (J30, 9 nov.) : DINOv2-B figé, régression logistique à trois
  classes (fond d'œil, dermoscopie, inconnu) et seuil de confiance 0,9 fixés avant le
  run. Test sur d'autres sources : 499/500 fonds d'œil (RFMiD), 499/500 dermoscopies
  (ISIC test), 500/500 photos ordinaires rejetées. Critère « aucune erreur croisée » non
  atteint (un fond d'œil très pathologique envoyé en dermoscopie). Non branché
  automatiquement dans `predict.py`. `docs/MODALITE.md`, `results/modality_router.json`,
  `scripts/train_modality_router.py` ; le placeholder MobileNet est remplacé. 4 tests.
- Dermatologie, contrôle ajouté après le run : 78 images du test ISIC ne sont jamais
  notées par le challenge (au moins une est un dessin) ; sans elles, ou sur les 5 920 images du score
  final officiel, PanDerm − DINOv2-L reste à +0,037 et +0,038.
- Défauts réels ODIR (9 nov.) : annotations techniques originales réassociées
  aux splits par patient, 24 préannotations visuelles provisoires du train
  conservées localement, étude de 463 images. Aucun gain de détection avec
  l'union luminosité/flou aux seuils fixés ; seulement 2 patients « low image
  quality » en validation. Résultats agrégés dans `results/quality_real_defects.json`,
  aucune nouvelle règle de rejet. Huit cas de test ajoutés ; 23 tests ciblés réussis.
- Diagnostic exploratoire d'exposition (J30) : `src/pipeline/exposure.py`,
  `scripts/quality_exposure_diagnostics.py` et `docs/QUALITE_EXPOSITION.md`.
  Premier calcul sur 120 images du train ODIR, cinq perturbations RGB fixées
  avant le run ; statistiques agrégées, sans nouveau seuil ni règle de rejet.
  Six cas de test ajoutés ; 15 tests ciblés exposition/qualité réussis.
- Résultats dermatologiques ISIC 2019 (9 nov.) : PanDerm-Large, DINOv2-L et
  DINOv2-B figés, même sonde linéaire et protocole défini avant le run.
  PanDerm − DINOv2-L : AUROC macro +0,037 [+0,030 ; +0,043] sur le test officiel.
  Résultats, gains par classe et réserves dans `docs/DERMATOLOGIE.md` ; JSON archivé,
  scores individuels conservés localement.
- Provenance des prochains rapports de neuf scripts d'analyse et de fine-tuning :
  commit, Python et versions des bibliothèques. Avec `--external-only`, provenance
  de l'évaluation et empreinte de l'artefact séparées du run d'entraînement.
  Deux cas de test ajoutés ; suite complète : 132 tests réussis le 9 novembre.
- Dermatologie, socle initial : `src/data/isic_dataset.py` (ISIC 2019, groupes par lésion),
  `src/models/timm_backbones.py` (DINOv2 ; Green y passe, résultats inchangés),
  `extract_embeddings.py --dataset isic2019 --backbone dinov2`. 3 tests.
- `load_embeddings` : chargeur d'embeddings commun aux analyses (configuration et
  complétude vérifiées), point restant de l'audit du 6 nov. Résultats inchangés. 1 test.
- `docs/DERMATOLOGIE.md` : choix des modèles (PanDerm, DINOv2, Derm Foundation, MONET)
  et des jeux ISIC pour la seconde indication, licences comprises.
- Notebooks : clonage possible d'un dépôt privé (secret `GITHUB_TOKEN`, jamais
  affiché) et révision du code fixable (`REVISION`), point restant de l'audit du 6 nov.
- Green à résolution égale : `RETFoundGreenBackbone` accepte une autre taille
  d'image (encodages de position interpolés, 392 px inchangé),
  `green_analysis.py --green-size 224`, `notebooks/kaggle_green_224.ipynb`. 1 test.
  Résultat du 8 nov. : EyePACS 0,680 contre 0,661 (+0,019 [−0,001 ; +0,038]),
  APTOS à format égal +0,014 [−0,004 ; +0,030], test RFMiD +0,002 : aucun gain
  démontré à 224 px. Fiche du modèle : « Green non servi » corrigé (audit du 6 nov.).
- RETFound à 384 px : `green_analysis.py --retfound-size 384`,
  `notebooks/kaggle_retfound_384.ipynb`. Résultat du 9 nov. : EyePACS 0,651 contre
  0,661 à 224 px (−0,010 [−0,020 ; −0,001]), APTOS 0,769 contre 0,827 ; Green à
  392 px reste devant (+0,075 [+0,053 ; +0,095] sur EyePACS).
- `docs/DROITS_DONNEES.md` : licences des modèles et des données (fond d'œil, ISIC),
  et points à trancher pour un usage en entreprise.

## [1.1.0] — 2026-11-06

### Ajouté
- Fine-tuning partiel de RETFound : `src/models/finetune.py` (4 derniers
  blocs et tête entraînables, partie figée sans gradient, augmentations
  avec masque commun réappliqué), `scripts/finetune_retfound.py`
  (configuration fixée à l'avance), `scripts/finetune_analysis.py`
  (comparaison appariée avec la tête linéaire, APTOS à format égal),
  `notebooks/colab_finetune_retfound.ipynb`. 4 tests. Résultats du 5 nov. :
  test RFMiD 0,873 (+0,026 [+0,014 ; +0,041] sur la tête linéaire) ; APTOS
  0,974 globale (+0,148) mais −0,012 [−0,026 ; +0,002] à format égal.
  `finetune_analysis.py` : bootstrap par groupes de doublons d'APTOS,
  différences appariées globale et à format égal, AUROC par format.
- Seconde validation externe, EyePACS : `src/data/eyepacs_dataset.py`
  (images extraites seulement, bootstrap par patient),
  `notebooks/colab_external_eyepacs.ipynb` (2 500 patients tirés au
  hasard). `extract_embeddings.py` accepte EyePACS et enregistre patient et
  format d'image ; `external_validation.py --external aptos|eyepacs`, plis
  du diagnostic groupés par patient. 2 tests.

- Résultats EyePACS (5 nov., 2 500 patients) : tête RFMiD 0,661
  [0,639 ; 0,683], ODIR 0,684, multi-source 0,675 ; à format égal 0,680 pour
  RFMiD ; tête entraînée dans EyePACS 0,785. `external_validation.py` ajoute
  l'AUROC à format égal et l'AUROC de chaque stade contre le stade 0.
  Réserve documentée : le pré-entraînement de RETFound inclut Kaggle EyePACS.
- Notebook EyePACS : téléchargement et extraction vérifiés, erreurs visibles.

- RETFound-Green : `src/models/retfound_green.py`,
  `extract_embeddings.py --backbone green` (et `--dataset rfmid`),
  `scripts/green_analysis.py` (comparaison appariée avec RETFound),
  `notebooks/colab_green_finetune_eyepacs.ipynb` et sa version Kaggle. 4 tests.
- `finetune_retfound.py --eyepacs-root`, `--external-only` (ex `--aptos-only`) ;
  `finetune_analysis.py` traite aussi EyePACS.

- Résultats du 6 nov. : fine-tuning sur EyePACS 0,650 contre 0,661
  (−0,012 [−0,031 ; +0,006]) ; RETFound-Green sur EyePACS 0,726 contre 0,661
  (+0,065 [+0,043 ; +0,085]), sur APTOS 0,958 contre 0,827, test RFMiD 0,871
  contre 0,846 (+0,025 [−0,001 ; +0,048]). Plafond dans EyePACS inchangé
  (0,786 contre 0,785). `notebooks/kaggle_green_finetune_eyepacs.ipynb`.

- `predict.py` : choix du modèle par la tête (RETFound ou RETFound-Green),
  `scripts/train_green_head.py`, `decision_metrics.py --variant`,
  `results/decision_rfmid_green.json`. Vérifié avec les vrais poids (écart
  ≤ 6×10⁻⁶ sur 20 images). 2 tests.

- Localisation sur IDRiD : `src/data/idrid_dataset.py`, `crop_mask_annotation`
  (même géométrie que `crop_mask`), `src/explainability/localization.py`,
  `scripts/localization_maps.py`, `scripts/localization_analysis.py`,
  `notebooks/kaggle_localization_idrid.ipynb`. 13 tests. Résultats : occlusion
  contre lésions, écart à la position +0,016 [−0,004 ; +0,039] pour RETFound,
  +0,083 [+0,055 ; +0,110] pour Green ; segmentation (AP, test) disque optique
  0,597 et 0,919, exsudats durs 0,425 et 0,651.

### Modifié
- Page GitHub : présentation, résumé en anglais, planche d'exemples, données et
  structure à jour. `LICENSE` réduit au texte officiel ; copyright et attributions
  dans `THIRD_PARTY_NOTICES.md`. GitHub affiche « Other » pour cette licence : voir
  le badge et le README.

### Corrigé (troisième audit, 6 novembre 2026)
- **Seuils** : liés à l'identité exacte de la tête (SHA-256 de ses poids) et à son
  contrat ; `decision_metrics.py --head` exige la preuve par les embeddings, en
  validation et en test. Une sonde seule laissait passer une tête aux poids
  différents. Fichiers `decision_*.json` régénérés, chiffres identiques.
- **Few-shot** : plis internes groupés, tirages archivés. Sensibilités et
  spécificités des têtes réentraînées inchangées à ±0,01.
- **Cache** : métadonnées adoptées marquées `assumed_not_verified` ; analyses Green
  contrôlées par leur configuration.
- **IDRiD** : dossiers et masques obligatoires refusés s'ils manquent, découpage
  officiel contrôlé, enrichissement sur exactement la surface annoncée (RETFound
  disque optique ×4,7 au lieu de ×4,5), cibles et probabilités du test archivées.
- **Fine-tuning** : `--external-only` refuse un artefact préparé autrement.
- **Provenance** : commit et versions dans les résultats régénérés ;
  `green_predictions.npz` et `localization_predictions.npz` permettent de recalculer.
- **Textes** : fiche du modèle structurée par variante ; « plafond » remplacé par la
  performance d'une sonde ; format présenté comme indicateur indirect de l'appareil.

### Corrigé (second audit, 5 novembre 2026)
- **Doublons d'images dans APTOS** : 122 groupes d'embeddings identiques
  (249 images, 6,8 %), dont 67 à cheval entre les moitiés « site » et
  « rapport ». `src/eval/duplicates.py` regroupe doublons exacts et quasi
  exacts (cosinus >= 0,999) avec les patients ; découpages, plis et
  bootstraps d'APTOS se font par ces groupes. Mesure externe inchangée
  (0,827) ; à format égal, la tête RFMiD passe de 0,942 à **0,919** et les
  têtes adaptées de 0,934-0,945 à 0,906-0,912 (moins bien que la tête
  RFMiD).
- Adaptation au site : seuil few-shot choisi hors échantillon (validation
  croisée interne sur les N images), plus sur les images d'ajustement.
- Seuils liés à la tête : sonde de tête enregistrée par
  `decision_metrics.py --head` (avec vérification sur la validation),
  refus par `predict.py` d'une autre tête ou d'un seuil hors [0, 1] ;
  `--allow-unverified-thresholds` pour un fichier historique.
- `--reject-unusable` : l'image refusée n'est plus passée dans le modèle.
- Cache d'embeddings : clés en double et valeurs non finies refusées ; les
  scripts d'analyse vérifient la configuration comme le pipeline principal.
- Occlusion : IC de la différence appariée par bootstrap sur les patients
  (DMLA −65 % [−77 ; −45]).
- Fine-tuning : artefact complet (taille d'image, prétraitement et sa
  version, empreinte du checkpoint de base, paramètres entraînés),
  `load_finetuned_state` avec test aller-retour ; comparaison refusant des
  classes ou des images différentes.
- Décisions : « assez de cas pour fixer un seuil » (validation) et « assez
  de cas pour l'évaluer » (test) distingués.
- Formulations : « format d'image d'origine, indicateur indirect de
  l'acquisition » au lieu d'« appareil » ; résultat à format égal présenté
  comme exploratoire, couverture par format publiée.
- CI : la vraie cause des échecs des tests lents était un disque plein
  (checkpoints aléatoires de 1,2 Go conservés par pytest) ; dossiers
  temporaires vidés après chaque exécution.

### Modifié
- `results/external_aptos.json` régénéré avec les champs génériques
  (`external_*`) ; chiffres identiques (diagnostic « à adresser » 0,966 ->
  0,967, plis désormais groupés).
- CI : un échec des tests lents s'affiche en annotation (nom du test et
  erreur) ; un fichier sans test lent sélectionné n'est plus une erreur.

## [1.0.0] — 2026-11-05

Première version taguée. Socle fond d'œil complet et évalué : corrections
issues de l'audit du 4 novembre 2026, contrôles de robustesse, métriques de
décision, prédiction sur image nouvelle, seuil de flou calibré, validation
externe sur APTOS 2019.

### Licence
- Dépôt sous **CC BY-NC 4.0** (`LICENSE`), cohérente avec RETFound, dont
  deux fichiers sont adaptés.

Corrections issues de l'audit du 4 novembre 2026. Résultats publiés
inchangés : prédictions identiques à 5·10⁻⁷ près (RFMiD et ODIR, validation
et test), étiquettes identiques.

### Corrigé
- Contrôle qualité : la résolution était mesurée après redimensionnement en
  224×224 (jamais trop petite) et le flou en convention BGR alors que les
  images sont en RGB. Il lit maintenant le fichier d'origine et mesure le
  flou sur la rétine à échelle fixe, sans le bord du disque.
- Cache d'embeddings : recherche des images manquantes quadratique en nombre
  d'images ; un checkpoint ou un prétraitement modifié sous le même nom
  réutilisait silencieusement des embeddings périmés (empreinte ajoutée).
- Chargement des poids : le repli sans `weights_only` (pickle) était
  automatique pour n'importe quel fichier. Il essaie d'abord en autorisant
  seulement `argparse.Namespace`, puis n'accepte le chargement complet que
  pour un fichier du dossier officiel ou avec `--trust-checkpoint`.
- Notebook : une erreur de `run_baseline.py` passait inaperçue (`| tee`
  renvoie le code de `tee`) ; chaque étape vérifie maintenant sa sortie.
- ODIR : une cellule de mots-clés vide devenait la classe O ; elle lève
  maintenant une erreur. Casse des mots-clés normalisée ; un nom de fichier
  sans `_left`/`_right` est refusé au lieu d'être pris pour l'œil droit.

### Modifié
- `--epochs` vaut 30 par défaut (réglage des résultats publiés, au lieu de 5).
- `--head-batch-size` (défaut 16) rend explicite la taille des mini-batchs
  de la tête ; `--batch-size` ne règle que l'extraction.
- JSON de résultats : métriques sous `eval` (`val` gardé pour
  compatibilité) et `manifest` (commit, empreinte du checkpoint, version du
  prétraitement, réglages, versions). Mêmes informations dans la tête
  sauvegardée.

### Ajouté
- `scripts/device_recalibration.py` et `results/device_recalibration.json` :
  recalibrage des scores (ou embeddings) par appareil, sans étiquette, sur
  le transfert RFMiD <-> ODIR. Pas de gain démontré (RFMiD -> ODIR : +0,003
  [-0,017 ; +0,022] au test ; le gain de validation ne se confirme pas) :
  entre ces bases, le transfert n'échoue pas à cause d'un décalage par
  appareil.
- `scripts/aptos_device_check.py`, `notebooks/colab_aptos_formats.ipynb`
  et `results/aptos_device_check.json` : contrôle à appareil égal sur
  APTOS (format d'image d'origine). Dans APTOS, l'appareil donne presque
  le diagnostic (lien 0,336) ; les têtes entraînées avec des images APTOS
  s'en servent (AUROC 0,98-0,99 globale, 0,934-0,964 à appareil égal) ; la
  tête RFMiD reconnaît la maladie à appareil égal (0,942) mais ses scores
  se décalent selon l'appareil (globale 0,830).
- `scripts/site_adaptation.py` et `results/site_adaptation_aptos.json` :
  adaptation à un nouveau site sur APTOS coupé en deux (site / rapport).
  Sans étiquette (centrage, CORAL) : pas de gain ; seuil recalibré avec 50
  images : sensibilité 0,91 mais spécificité 0,39 ; tête réentraînée avec
  50 images : AUROC 0,981, gain suspect à cause du biais d'acquisition
  d'APTOS (analyse de structure ajoutée à `external_validation.py`).
- `results/external_aptos.json` : validation externe de la rétinopathie
  diabétique sur APTOS (3 662 images). AUROC 0,827 (tête RFMiD, 0,947 sur
  son test), 0,526 (ODIR), 0,738 (multi-source) ; seuils non
  transposables (sensibilité 0,51 au lieu de 0,90) ; diagnostic : 0,995
  avec une tête entraînée dans APTOS.
- Validation externe sur APTOS 2019 (rétinopathie diabétique) :
  `src/data/aptos_dataset.py`, `scripts/extract_embeddings.py` (GPU),
  `scripts/external_validation.py` (CPU, seuils hérités de la validation
  d'origine) et `notebooks/kaggle_external_aptos.ipynb`.
- `scripts/calibrate_quality.py` et `results/quality_calibration.json` :
  seuil de flou calibré à 4,16 (2e percentile des images d'entraînement
  ODIR acceptables), désormais par défaut (`DEFAULT_BLUR_THRESHOLD`) dans
  `run_baseline.py` et `predict.py`. Détecte 79 à 90 % du flou franc simulé,
  signale 1,9 % d'ODIR et 0,1 % de RFMiD ; ne détecte pas les 21 images
  « low image quality » d'ODIR (AUROC 0,51), dont le défaut n'est pas le
  flou ; perte du modèle indépendante de la netteté. `openpyxl` en
  dépendance de développement (annotations d'origine d'ODIR).
- `scripts/robustness_checks.py` et `results/robustness_checks.json` :
  appareil retrouvé à partir des embeddings, AUROC à appareil égal
  (`stratification_gap` dans `src/eval/metrics.py`), transfert entre RFMiD
  et ODIR pour la rétinopathie diabétique, la DMLA et la myopie.
- `results/occlusion_population_odir.json` : occlusion sur échantillon tiré
  au hasard (ODIR, Kaggle T4) ; DMLA hors rétine 77 % → 12 % (apparié −65 %
  [−75 ; −50]). `results/predict_check_odir.json` : `predict.py` avec les
  vrais poids redonne les probabilités de l'entraînement (écart ≤ 10⁻⁶).
  `results/results_odir_{none,crop-mask}.json` régénérés avec manifeste
  (mêmes AUROC).
- `scripts/occlusion_population.py` : occlusion sur un échantillon tiré au
  hasard (malades et sains, erreurs comprises), mêmes images pour plusieurs
  têtes, différence appariée ; `notebooks/kaggle_gpu_checks.ipynb` l'enchaîne
  sur ODIR (`none` contre `crop-mask`) avec le contrôle de `predict.py`
  sur les vrais poids.
- `predict.py` et `src/pipeline/inference.py` : prédiction sur une image
  nouvelle avec la même chaîne que l'entraînement (contrôle qualité,
  prétraitement de la tête, RETFound, tête), décision par maladie avec les
  seuils de la validation, refus d'une tête incompatible (checkpoint ou
  version du prétraitement), option `--reject-unusable`.
- `src/eval/decision.py`, `scripts/decision_metrics.py` et
  `results/decision_{rfmid,odir}.json` : seuils par maladie choisis sur la
  validation (F1 maximal ou sensibilité ≥ 0,90), sensibilité, spécificité,
  VPP, VPN, faux positifs et négatifs au test, précision moyenne, Brier,
  ECE, courbes de fiabilité et température ; maladies à moins de 10
  positifs marquées « preuve insuffisante ».
- `scripts/domain_remedies.py` et `results/domain_remedies.json` : remèdes
  comparés sur la validation (normalisation par base, effacement linéaire
  de l'appareil LEACE, tête multi-source) ; tête multi-source retenue et
  lue une fois sur le test.
- `scripts/sweep_head.py` et `results/sweep_head_{rfmid,odir}.json` :
  archive du balayage des réglages de la tête (train et validation
  seulement).

## [0.1.0] — 2026-11-04

Jamais taguée : intégrée dans la 1.0.0. Premier jalon : socle fond d'œil de bout en bout avec les vrais poids RETFound.

### Ajouté
- `run_baseline.py`, point d'entrée unique : contrôle qualité, embeddings
  RETFound figés sur GPU, tête multi-label, AUROC et F1 macro, gap
  train − validation, graine fixée (`--seed`), export JSON (`--results-json`).
- Loaders RFMiD et ODIR sur la structure Kaggle réelle ; étiquettes ODIR par
  œil (`--odir-labels`), split par patient.
- Script Grad-CAM sur les prédictions de la tête entraînée
  (`scripts/gradcam_examples.py`, planches pas encore produites), sauvegarde
  de la tête (`--save-head`).
- `--eval-split test` : évaluation sur le jeu de test officiel RFMiD.
- Intervalles de confiance à 95 % des AUROC (bootstrap sur les patients),
  nombre de positifs par classe, alerte sous 10 positifs.
- `--predictions` : probabilités image par image ; `scripts/compare_predictions.py`
  compare deux runs par différence d'AUROC appariée.
- Split ODIR 70/15/15 par patient (`--odir-test-fraction`, `--eval-split test`
  pour ODIR) : validation inchangée, test tiré dans l'ancien train.
- Cache d'embeddings (`--embeddings-cache`) : le backbone n'est chargé que
  pour les images absentes du cache.
- Notebook : compare `none` et `crop-mask` en un run, cache sur Google Drive,
  test final protégé par `RUN_TEST`.
- Prétraitement commun `--preprocess crop-mask` : recadrage sur la rétine et
  masque identique pour toutes les images, contre le raccourci d'acquisition
  (format d'image lié à la maladie dans ODIR).
- Test d'occlusion (`scripts/occlusion_examples.py`, cellule Colab 5 bis) :
  contrôle des cartes Grad-CAM sur les mêmes images, avec mesure de
  l'effet du fond.
- Notebook Google Colab (poids via `HF_TOKEN`, données via kagglehub).
- Résultats finaux (`crop-mask`, test lu une seule fois) : AUROC macro
  **0,846** sur le test RFMiD (validation 0,864) et **0,848** sur le test
  ODIR (validation 0,861). Premiers résultats avec `none` : RFMiD 0,858
  (validation) et 0,832 (test) ; ODIR 0,845 avec étiquettes par œil, 0,806
  avec étiquettes patient (split 85/15).
- Notebook utilisable sur Kaggle comme sur Colab (secrets, cache local, zip
  des sorties).
- CI (lint ruff puis tests), `pyproject.toml`, `pre-commit`, carte du modèle,
  guide d'utilisation, test du classifieur de modalité.

### Modifié
- `--preprocess crop-mask` devient le défaut : sur la validation, il fait
  +0,020 d'AUROC sur ODIR (IC [+0,011 ; +0,030]), sans coût sur RFMiD, et
  ramène l'effet d'occlusion de la DMLA d'ODIR sur la rétine (92 % → 5 %
  hors champ). `--preprocess none` reproduit les premiers chiffres.

### Corrigé
- Planche d'occlusion : le « Δ » s'affichait en carré (absent de la police).
- La loss affichée à chaque epoch est la moyenne de l'epoch, plus celle du
  dernier batch.
- Occlusion avec `crop-mask` : champ rétinien = masque commun exact, plus un
  seuillage qui comptait les zones sombres de la rétine comme du fond.
- `scripts/download_rfmid_odir.py` copie ou relie les données (kagglehub) au
  lieu de déplacer les fichiers source.
- Un checkpoint RETFound auquel il manque des poids de l'encodeur était
  chargé en silence (couches aléatoires) : il est maintenant refusé.
- Le contrôle qualité annonçait des images « rejetées » qui restaient dans
  le jeu : il les signale désormais comme telles, sans filtrer.
- Normalisation ImageNet manquante avant RETFound.
- Plantage sur GPU (tête sur `cuda`, embeddings de validation sur CPU).
- Grad-CAM inopérant sur backbone gelé (entrée non dérivable).

### Retiré
- `scripts/smoke_test.py`, doublon du mode synthétique de `run_baseline.py`.
- Évaluation AUROC dupliquée dans `run_baseline.py`, remplacée par
  `src/eval/metrics.py`.
- Dépendances inutilisées (`albumentations`, `pyyaml`, `tqdm`).
