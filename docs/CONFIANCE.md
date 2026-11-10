# Score de confiance : choisir un compromis par maladie (J60)

Au 10 novembre 2026, les probabilités sont bien calibrées en interne (ECE ≈ 0,02-0,03,
rapports `results/decision_*.json`). Reste à choisir, pour chaque maladie, **combien de cas
manqués et de fausses alertes on accepte** : c'est une décision d'usage, pas de modèle.
Cette page donne les compromis possibles pour aider ce choix.

Méthode (`scripts/operating_points.py`, plan fixé avant le calcul, commit `ee9c3b6`) :
seuil choisi sur la validation pour viser une sensibilité de 80, 90 ou 95 %, appliqué au
test ; résultats ramenés à **1 000 personnes**, avec la proportion de malades du test.
Le test a déjà servi : chiffres descriptifs, pas une confirmation. Proportions de malades
bien plus élevées que dans une population dépistée : dans la vraie vie, à sensibilité
égale, les fausses alertes pèseraient davantage. Résultats complets :
`results/operating_points_odir_crop-mask.json` et `results/operating_points_rfmid_green.json`.

## ODIR (RETFound, 965 images de test)

| Maladie | Malades / 1 000 | Sensibilité visée 80 % : manqués / fausses alertes | 90 % | 95 % |
|---|---|---|---|---|
| Rétinopathie diabétique (D) | 276 | 56 / 304 | 20 / 442 | 3 / 540 |
| Glaucome (G) | 35 | 6 / 134 | 1 / 309 | 0 / 478 |
| Cataracte (C) | 48 | 16 / 16 | 15 / 22 | 8 / 58 |
| DMLA (A) | 48 | 18 / 121 | 6 / 283 | 4 / 462 |
| Myopie (M) | 38 | 3 / 5 | 1 / 10 | 1 / 11 |
| Hypertension (H) | 18 | 6 / 344 | 0 / 802 | 0 / 805 |

## RFMiD (Green, 640 images de test), principales maladies

| Maladie | Malades / 1 000 | 80 % : manqués / fausses alertes | 90 % | 95 % |
|---|---|---|---|---|
| Rétinopathie diabétique (DR) | 194 | 48 / 33 | 20 / 98 | 8 / 209 |
| DMLA (ARMD) | 48 | 17 / 53 | 14 / 77 | 11 / 216 |
| Myopie (MYA) | 50 | 11 / 2 | 2 / 33 | 2 / 56 |
| Trou maculaire (MH) | 162 | 14 / 66 | 6 / 109 | 0 / 167 |

## Ce qu'on en retient

- **Myopie et cataracte** : peu de fausses alertes même à haute sensibilité ; seuils faciles
  à fixer.
- **Rétinopathie diabétique** : sur RFMiD avec Green, 90 % de sensibilité coûte environ
  100 fausses alertes pour 1 000 ; sur ODIR avec RETFound, plus de 400. Le compromis dépend
  fortement de la base et du modèle, et un seuil ne se transfère pas d'un site à l'autre
  (sensibilité de 13 % sur EyePACS au seuil de RFMiD).
- **DMLA, glaucome, hypertension** : viser 90 % multiplie les fausses alertes (283, 309 et
  802 pour 1 000 sur ODIR) ; l'hypertension n'est pas exploitable telle quelle.
- **À décider avec le tuteur** : l'usage (dépistage large, aide au spécialiste), qui fixe
  la sensibilité visée par maladie. Le seuil devra ensuite être recalibré et confirmé sur le
  site ou la cohorte d'usage.
