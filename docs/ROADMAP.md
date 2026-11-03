# Roadmap Visual Health

## J30 — Socle
- [x] Contrôle qualité (flou/cadrage) opérationnel (seuil de flou encore à calibrer)
- [x] Backbone RETFound intégré (embeddings figés)
- [x] Premier classifieur fond d'œil sur RFMiD/ODIR, première AUROC (RFMiD 0,855 / ODIR 0,806 en étiquettes patient, 2026-11-02 ; ODIR en étiquettes par œil à mesurer)
- [ ] Grad-CAM fonctionnel sur les prédictions (code prêt : `scripts/gradcam_examples.py` + cellule du notebook Colab ; reste à le lancer avec les vrais poids et à relire les cartes)

## J60 — Localisation
- [ ] Localisation/segmentation des zones suspectes (ex. disque optique)
- [ ] Seconde indication amorcée (dermatologie via ISIC)
- [ ] Score de confiance calibré

## J90 — Modularité démontrée
- [ ] Socle modulaire validé sur 2 modalités (rétine + dermato)
- [ ] Dossier d'accès AI-Hub (étude TSA/rétine) avancé
- [ ] Note de synthèse pour arbitrage de la 3e indication
