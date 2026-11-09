# Consignes Visual Health

## Commits locaux

- Autorisation permanente de Marc : commiter le travail terminé et vérifié pour
  conserver une trace des changements. Ne jamais faire de push sans demande explicite.
- Conserver l'identité `Marc Su <sumarc183@gmail.com>`, sans mention de co-auteur.
- Le soir et le week-end, préparer les changements sans commiter ; proposer le
  commit au jour ouvré suivant. Ne jamais antidater.
- Nommer chaque fichier dans `git add`, vérifier l'index et `git status` avant le
  commit. Ne jamais utiliser `git add .`, `git add -A` ou `git commit -a`.
- Ne jamais commiter `docs/AUDIT.md`, les caches d'embeddings, les
  `results/green_predictions_*.npz`, ni les étiquettes ou identifiants APTOS/EyePACS.
- `results/dermatology_isic2019.npz` reste local sans accord explicite de Marc.

## Périmètre

- Travailler uniquement sur Visual Health ; ne jamais modifier SleepFM.
- Ne modifier le journal ou la documentation Claude que sur demande de Marc.
- Jalons fixes : J30 le 1er décembre 2026, J60 le 31 décembre 2026,
  J90 le 30 janvier 2027.
