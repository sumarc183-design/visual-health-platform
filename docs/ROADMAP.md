# Roadmap Visual Health

## J30 — Socle

> Atteint au sens « première baseline » (4 nov.). Pas encore un socle
> opérationnel : contrôle qualité non calibré, routage de modalité non
> entraîné (audit du 4 novembre). Commande de prédiction sur image nouvelle
> (`predict.py`) livrée le 4 novembre, pas encore lancée avec les vrais poids.

- [x] Contrôle qualité (flou, résolution) implémenté : sonde sur un échantillon, sans filtrage ; sur le fichier d'origine depuis le 4 nov. ; seuil de flou encore à calibrer
- [x] Backbone RETFound intégré (embeddings figés)
- [x] Premier classifieur fond d'œil sur RFMiD/ODIR, première AUROC (2026-11-03, tête figée : RFMiD 0,858 en validation et 0,832 sur le test officiel ; ODIR 0,845 en étiquettes par œil, 0,806 en étiquettes patient)
- [x] Explicabilité sur les vraies prédictions (2026-11-03) : Grad-CAM peu fiable sur ce ViT, remplacé par l'occlusion ; raccourci d'acquisition détecté sur la DMLA d'ODIR
- [x] Protocole d'évaluation (2026-11-03) : IC à 95 % par bootstrap (patients pour ODIR, images pour RFMiD), prédictions sauvegardées, comparaison appariée entre variantes, test ODIR mis de côté (70/15/15), cache d'embeddings
- [x] Prétraitement commun `crop-mask` (2026-11-04) : +0,020 d'AUROC sur ODIR (significatif), neutre sur RFMiD, raccourci DMLA réduit sur les exemples examinés (92 % → 5 % de l'effet hors rétine) ; test final : RFMiD 0,846, ODIR 0,848
- [x] Tête (2026-11-04) : balayage de 24 configurations × 6 durées (régularisation, pondération des classes rares, standardisation, 5 à 50 epochs), sur la validation seulement : pas de gain démontré (+0,003 RFMiD, +0,007 ODIR au mieux, maxima biaisés vers le haut). Archivé : `scripts/sweep_head.py`, `results/sweep_head_*.json`. Tête inchangée

## J60 — Localisation
- [ ] Localisation/segmentation des zones suspectes (ex. disque optique)
- [ ] Seconde indication amorcée (dermatologie via ISIC)
- [ ] Score de confiance calibré (avancé le 4 nov. : probabilités déjà bien calibrées, ECE ≈ 0,02-0,03, seuils par maladie choisis sur la validation ; reste la validation clinique du compromis)

## J90 — Modularité démontrée
- [ ] Socle modulaire validé sur 2 modalités (rétine + dermato)
- [ ] Dossier d'accès AI-Hub (étude TSA/rétine) avancé
- [ ] Note de synthèse pour arbitrage de la 3e indication
