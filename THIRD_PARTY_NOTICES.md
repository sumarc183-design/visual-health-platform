# Attributions et licences

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

## Jeux de données

- **RFMiD** et **ODIR-5K** : téléchargés depuis Kaggle (miroirs publics des
  challenges RIADD et ODIR-2019). Respecter les conditions d'usage propres à
  chaque jeu ; les images ne sont pas versionnées dans ce dépôt.
