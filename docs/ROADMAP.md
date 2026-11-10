# Roadmap Visual Health

État au 9 novembre 2026. Échéances fixes : J30 le 1er décembre 2026,
J60 le 31 décembre 2026 et J90 le 30 janvier 2027.

## J30 — Socle (1er décembre 2026)

> Atteint au sens « première baseline » (4 nov.). Pas encore un socle
> opérationnel : contrôle qualité limité au flou et à la résolution (seuil
> de flou calibré le 5 nov.), routage de modalité non entraîné à la date de l'audit du
> 4 novembre (entraîné et évalué depuis le 9 nov., voir plus bas). Commande de prédiction sur image nouvelle
> (`predict.py`) livrée et vérifiée avec les vrais poids le 4 novembre.

- [x] Contrôle qualité (flou, résolution) implémenté : sonde sur un échantillon, sans filtrage ; sur le fichier d'origine depuis le 4 nov. ; seuil de flou calibré le 5 nov. (4,16 : flou franc détecté à 79-90 %, ~2 % d'ODIR signalé ; ne détecte pas les autres défauts de qualité)
- [x] Backbone RETFound intégré (embeddings figés)
- [x] Premier classifieur fond d'œil sur RFMiD/ODIR, première AUROC (2026-11-03, tête figée : RFMiD 0,858 en validation et 0,832 sur le test officiel ; ODIR 0,845 en étiquettes par œil, 0,806 en étiquettes patient)
- [x] Explicabilité sur les vraies prédictions (2026-11-03) : Grad-CAM peu fiable sur ce ViT, remplacé par l'occlusion ; raccourci d'acquisition détecté sur la DMLA d'ODIR
- [x] Protocole d'évaluation (2026-11-03) : IC à 95 % par bootstrap (patients pour ODIR, images pour RFMiD), prédictions sauvegardées, comparaison appariée entre variantes, test ODIR mis de côté (70/15/15), cache d'embeddings
- [x] Prétraitement commun `crop-mask` (2026-11-04) : +0,020 d'AUROC sur ODIR (significatif), neutre sur RFMiD, raccourci DMLA réduit sur les exemples examinés (92 % → 5 % de l'effet hors rétine) ; test final : RFMiD 0,846, ODIR 0,848
- [x] Tête (2026-11-04) : balayage de 24 configurations × 6 durées (régularisation, pondération des classes rares, standardisation, 5 à 50 epochs), sur la validation seulement : pas de gain démontré (+0,003 RFMiD, +0,007 ODIR au mieux, maxima biaisés vers le haut). Archivé : `scripts/sweep_head.py`, `results/sweep_head_*.json`. Tête inchangée
- [ ] Contrôle qualité au-delà du flou et de la résolution : définir des défauts
  annotés et mesurer leur détection avant d'ajouter des règles de rejet.
  Diagnostic descriptif d'exposition préparé et exécuté le 9 novembre sur
  120 images du train ODIR ; [protocole et limites](QUALITE_EXPOSITION.md).
  Étude de 463 images réelles et 24 préannotations visuelles provisoires :
  aucun gain de détection avec l'union luminosité/flou aux seuils fixés ;
  seulement 2 patients « low image quality » en validation. Relecture humaine
  et annotations spécifiques encore nécessaires ; aucune nouvelle règle de rejet.
  Relecture humaine faite le 9 novembre (36 images, série équilibrée) : les mots-clés
  d'ODIR concordent avec ce qu'un humain voit. Détecteur appris sur défauts simulés :
  aucun gain démontré sur les vrais défauts.
  Détecteur appris sur de vrais défauts (10 nov.) : repère la poussière, pas les
  défauts généraux, faute d'exemples ; il faut plus d'images annotées.
  Appris sur les notes d'experts de DeepDRiD (10 nov.) : repère 67 % des images de
  qualité insuffisante (5 % pour le détecteur actuel) et la plupart des défauts graves,
  mais avec beaucoup de fausses alertes ; candidat pour un avertissement, pas un rejet.
- [ ] Identification de modalité : routeur entraîné et évalué le 9 novembre
  (DINOv2-B figé, plan fixé avant le run) : 499/500 fonds d'œil, 499/500 dermoscopies
  et 500/500 photos ordinaires bien routés sur d'autres sources. Critère « aucune
  erreur croisée » non atteint (un fond d'œil pathologique envoyé en dermoscopie) ;
  intégration à `predict.py` comme garde-fou à décider avec le tuteur
  ([détails](MODALITE.md)). Protocole 2 (10 nov.) : aucune des 756 images médicales
  jamais vues (OCT, échographie, scanner) mal routée ; même erreur croisée résiduelle.

## J60 — Localisation (31 décembre 2026)
- [x] Localisation/segmentation des zones suspectes (6 nov., première mesure sur IDRiD) : occlusion contre masques d'experts, segmentation par case sur tokens de patch, RETFound et Green ; disque optique AP 0,919, exsudats durs 0,651 (Green, 27 images de test). Pas de validation externe ni d'intégration à `predict.py`.
- [x] Seconde indication amorcée (9 nov., dermatologie sur ISIC 2019) : PanDerm,
  DINOv2-L et DINOv2-B figés ; gain principal d'AUROC macro de PanDerm sur
  DINOv2-L +0,037 [+0,030 ; +0,043] sur le test officiel. Sonde linéaire uniquement,
  recouvrement de préentraînement possible ; [résultats et réserves](DERMATOLOGIE.md#résultats).
- [ ] Score de confiance calibré (avancé le 4 nov. : probabilités déjà bien calibrées, ECE ≈ 0,02-0,03, seuils par maladie choisis sur la validation ; reste la validation clinique du compromis)
  Compromis par maladie préparés le 10 nov. pour aider ce choix ([CONFIANCE.md](CONFIANCE.md)).

- [ ] Protocole de confirmation daté avant toute nouvelle lecture (audit du 6 nov.) : variantes figées, et lot de patients EyePACS encore jamais examinés. Les bases actuelles sont exploratoires.

## J90 — Modularité démontrée (30 janvier 2027)
- [ ] Socle modulaire validé sur 2 modalités (rétine + dermato)
- [ ] Dossier d'accès AI-Hub (étude TSA/rétine) avancé
- [ ] Note de synthèse pour arbitrage de la 3e indication
