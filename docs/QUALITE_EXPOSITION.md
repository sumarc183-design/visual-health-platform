# Contrôle qualité : diagnostic d'exposition

Protocole exploratoire fixé le 9 novembre 2026, avant son premier calcul.
Objectif J30 : ajouter des mesures descriptives pour préparer l'étude des défauts
d'exposition, sans modifier le seuil de flou ou les décisions du pipeline.

## Protocole

- Source : images originales du découpage d'entraînement ODIR déjà utilisé par
  le projet, jusqu'à 120 images tirées sans remise, graine 0. Aucune image de
  validation ou de test n'est ouverte.
- Géométrie commune : `crop-mask`, 512 px, masque intérieur `retina_blur_mask`.
  Le fond noir et le bord du champ sont exclus des statistiques.
- Perturbations fixées : multiplier les canaux RGB par 0,25, 0,5, 1, 1,5 et 2,
  puis écrêter à 255. Le recadrage et le masque sont fixés avant la perturbation.
  C'est une simulation numérique, pas une modélisation physique de l'exposition.
- Mesures : luminance moyenne et médiane sur l'échelle 0–255, fraction des pixels
  de luminance ≤ 10 et ≥ 245. Ces niveaux définissent des statistiques de pixels ;
  ce ne sont pas des seuils de rejet d'images.
- Sortie : médiane et quartiles de chaque mesure par perturbation, commit et
  versions, empreinte SHA-256 de l'échantillon d'entrée. Aucun identifiant,
  annotation ou score individuel n'est publié.
- Aucun seuil n'est ajusté, aucune performance diagnostique n'est recalculée,
  aucune image n'est retirée de l'entraînement ni du parcours de prédiction.

Implémentation : `src/pipeline/exposure.py` et
`scripts/quality_exposure_diagnostics.py`.

```bash
"../.venv-audit/bin/python" scripts/quality_exposure_diagnostics.py \
    --odir-root data/raw/odir --out results/quality_exposure_diagnostics.json
```

## Critères de portée

La sensibilité de ces mesures à des perturbations synthétiques est un contrôle
technique. Elle ne prouve pas la détection de sous-exposition, surexposition,
cataracte ou autre défaut sur des images réelles. Aucune annotation réelle de
qualité n'est utilisée par ce diagnostic.

À faire : constituer des annotations explicites de défauts réels, définir une
calibration sur le train puis une évaluation distincte par patient, mesurer les
fausses alertes et la détection avant d'envisager une nouvelle règle de rejet.
La licence d'ODIR reste à clarifier, comme décrit dans
[DROITS_DONNEES.md](DROITS_DONNEES.md).

## Premier calcul (9 novembre 2026)

Exécuté sur 120 images du train, avec le code du commit `c7f8155`, sans
modification suivie au moment du calcul. Résultats complets :
[quality_exposure_diagnostics.json](../results/quality_exposure_diagnostics.json).

| Multiplicateur RGB | Luminance moyenne par image : médiane [Q1 ; Q3] |
|---|---|
| 0,25 | 23,5 [18,7 ; 29,8] |
| 0,5 | 47,0 [37,4 ; 59,5] |
| 1 (original) | 94,0 [74,8 ; 119,1] |
| 1,5 | 140,3 [110,4 ; 173,6] |
| 2 | 174,6 [143,1 ; 206,8] |

Les quartiles décrivent la dispersion entre les images, pas des intervalles de
confiance. La luminance suit le sens des perturbations, comme attendu par
construction. La fraction de pixels sombres reste nulle pour l'image médiane,
même après multiplication par 0,25 ; une image assombrie ne contient pas
nécessairement de pixels proches du noir. À ×2, la fraction médiane de pixels
très clairs est de 0,66 % [Q1 0,05 % ; Q3 8,33 %]. Ces seules fractions ne
constituent donc pas un détecteur général des défauts d'exposition.

Contrôles exécutés : 15 tests ciblés (exposition et contrôle qualité) réussis,
lint et format vérifiés ; sortie agrégée cohérente avec les cinq perturbations.
La suite complète précédente comptait 132 tests réussis avant l'ajout de cette
brique ; elle n'a pas été relancée depuis. Le diagnostic n'améliore pas encore
la détection mesurée de défauts réels : les annotations et l'évaluation distincte
restent à faire.

## Défauts réels : protocole exploratoire du 9 novembre

Plan fixé avant le calcul des scores réels, dans `scripts/quality_real_defects.py`.
Les annotations techniques viennent du `data.xlsx` original, y compris les yeux
retirés de `full_df.csv`. Les patients gardent leur split du projet. Les patients
du test et ceux absents du CSV nettoyé sont exclus ; leurs images ne sont pas ouvertes.

- Tous les yeux avec un des six mots-clés techniques source sont conservés ;
  120 yeux témoins sont tirés par split (graine 0). Un patient avec un défaut
  connu ne peut pas devenir témoin par l'autre œil.
- Critère principal : annotation source « low image quality ». Secondaire :
  « lens dust ». Ces mots-clés ne décrivent pas spécifiquement l'exposition ;
  leur absence ne certifie pas une image de bonne qualité.
- Nouvelle alarme exploratoire : luminance moyenne sous le percentile 2 des
  témoins du train. Comparateur : seuil de flou existant 4,16. Leur union est
  rapportée sans réglage supplémentaire. Le pipeline de prédiction reste inchangé.
- Évaluation sur les patients de validation : patient signalé si au moins un
  œil sélectionné est signalé. Sensibilité et fausses alertes ont des intervalles
  binomiaux exacts à 95 %. L'AUROC est calculée par image, avec bootstrap par
  patient (500 tirages) ; aucun IC d'AUROC si une classe compte moins de 10 patients.
- Les comptes source disponibles annoncent seulement 2 yeux « low image quality »
  en validation : aucune conclusion solide ne pourra être tirée sur ce critère.
  L'échantillon est enrichi en défauts ; il n'estime pas la prévalence réelle.
- Une revue visuelle de 24 images du train utilise des codes anonymes, sans
  affichage des mots-clés ni des scores. Les préannotations techniques sont
  provisoires, sans avis clinique, et restent sous `data/quality_review/`, hors Git.
  Elles ne servent ni à calibrer le seuil ni à évaluer la validation.

Préparation locale :

```bash
"../.venv-audit/bin/python" scripts/quality_real_defects.py \
    --odir-root data/raw/odir --prepare-only
```

Calcul après revue, sans sélection de variante selon le résultat : même commande
sans `--prepare-only`. Le JSON ne contient que des agrégats et les empreintes
de provenance ; images, mots-clés par œil, mesures individuelles et préannotations
restent locaux. Cette analyse reste exploratoire : les mots-clés et la validation
ODIR ont déjà été examinés dans des travaux antérieurs du projet.

## Résultats sur les défauts réels (9 novembre)

Code exécuté : `0e09729`, sans modification suivie. Analyse de 463 images :
303 du train (dont 120 témoins) et 160 de validation (dont 120 témoins,
110 patients). Les 646 patients du test ou absents du CSV nettoyé sont exclus.
Source : [quality_real_defects.json](../results/quality_real_defects.json).

Le seuil de luminance moyenne appris sur les 120 témoins du train vaut
**49,108 sur l'échelle 0–255**. Il n'est pas intégré au pipeline.
Les taux ci-dessous portent sur les patients, signalés si au moins un œil
sélectionné déclenche l'alarme ; intervalles exacts à 95 %.

| Alarme | « Low image quality » : 2 patients | « Lens dust » : 26 patients | Fausses alertes : 110 patients témoins |
|---|---|---|---|
| Luminance basse | 0/2 : 0 % [0 ; 84,2] | 1/26 : 3,8 % [0,1 ; 19,6] | 1/110 : 0,9 % [0,02 ; 5,0] |
| Flou existant | 0/2 : 0 % [0 ; 84,2] | 3/26 : 11,5 % [2,4 ; 30,2] | 0/110 : 0 % [0 ; 3,3] |
| Union des deux | 0/2 : 0 % [0 ; 84,2] | 3/26 : 11,5 % [2,4 ; 30,2] | 1/110 : 0,9 % [0,02 ; 5,0] |

**Aucune détection supplémentaire avec l'union aux seuils fixés**, mais une
fausse alerte de plus sur les témoins. Deux cas positifs ne permettent ni de
conclure à une sensibilité nulle dans la population, ni de valider un détecteur.
Les témoins n'étant pas certifiés sans défaut, ces fausses alertes sont définies
par rapport aux mots-clés source, pas à une adjudication experte.

Sur les 35 images « lens dust » de validation, l'AUROC par image vaut 0,530
[0,408 ; 0,665] pour la luminance basse et 0,756 [0,638 ; 0,856] pour le flou
(bootstrap par patient). Sur les 2 images « low image quality », les estimations
ponctuelles sont respectivement 0,504 et 0,900, sans IC selon le protocole :
l'effectif est trop faible pour une interprétation robuste. Aucune supériorité
statistique entre détecteurs n'est déduite de leurs intervalles séparés.

### Revue visuelle et suite

Les 24 images anonymisées du train ont reçu des préannotations techniques par
une revue visuelle automatisée : visibilité du fond d'œil, assombrissement,
zones très claires, voile/flou possible et artefacts. Deux vues externes de
l'œil ont été repérées. Les cas ambigus sont notés `uncertain`, et les mesures
d'exposition sont `not_applicable` lorsque le champ rétinien est absent.
Il ne s'agit pas d'annotations cliniques validées ni d'une seconde lecture humaine.

Les planches, la grille `review.csv`, la correspondance des codes et les mesures
individuelles sont sous `data/quality_review/`, hors Git. L'empreinte de la grille
et le compte des 24 préannotations provisoires sont enregistrés dans le JSON.
Ces préannotations n'ont pas été utilisées pour les performances ci-dessus.

Contrôles : 23 tests ciblés réussis ; lint et format vérifiés ; contrôle
indépendant du seuil train, de la disjonction des patients, des AUROC par les
rangs, des comptes d'alarmes par patient et de l'empreinte de la grille locale.
La suite complète n'a pas été relancée pour cette brique indépendante.

À faire : relecture humaine des préannotations et annotations spécifiques des
défauts sur un échantillon plus large, avec plusieurs lecteurs si possible.
Tout futur détecteur doit avoir son protocole fixé avant une nouvelle mesure.
Le résultat actuel ne justifie aucune nouvelle règle de rejet ; les défauts de
modalité (vue externe de l'œil) doivent aussi être traités séparément de l'exposition.
