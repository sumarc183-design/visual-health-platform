# Roadmap Visual Health

## J30 — Socle
- [x] Contrôle qualité (flou/cadrage) opérationnel (seuil de flou encore à calibrer)
- [x] Backbone RETFound intégré (embeddings figés)
- [x] Premier classifieur fond d'œil sur RFMiD/ODIR, première AUROC (2026-11-03, tête figée : RFMiD 0,858 en validation et 0,832 sur le test officiel ; ODIR 0,845 en étiquettes par œil, 0,806 en étiquettes patient)
- [x] Explicabilité sur les vraies prédictions (2026-11-03) : Grad-CAM peu fiable sur ce ViT, remplacé par l'occlusion ; raccourci d'acquisition détecté sur la DMLA d'ODIR
- [x] Protocole d'évaluation (2026-11-03) : IC à 95 % par bootstrap sur les patients, prédictions sauvegardées, comparaison appariée entre variantes, test ODIR mis de côté (70/15/15), cache d'embeddings
- [x] Prétraitement commun `crop-mask` (2026-11-04) : +0,020 d'AUROC sur ODIR (significatif), neutre sur RFMiD, raccourci DMLA supprimé (92 % → 5 % de l'effet hors rétine) ; test final : RFMiD 0,846, ODIR 0,848
- [ ] Tête : pondération des classes rares, weight decay, arrêt précoce (écart train − test +0,14 sur RFMiD), à régler sur val grâce aux embeddings en cache (`embeddings/`, CPU suffit)

## J60 — Localisation
- [ ] Localisation/segmentation des zones suspectes (ex. disque optique)
- [ ] Seconde indication amorcée (dermatologie via ISIC)
- [ ] Score de confiance calibré

## J90 — Modularité démontrée
- [ ] Socle modulaire validé sur 2 modalités (rétine + dermato)
- [ ] Dossier d'accès AI-Hub (étude TSA/rétine) avancé
- [ ] Note de synthèse pour arbitrage de la 3e indication
