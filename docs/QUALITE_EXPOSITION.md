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
