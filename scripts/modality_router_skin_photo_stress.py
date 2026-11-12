#!/usr/bin/env python3
"""Routeur de modalité face à des photos de peau sans dermatoscope (SLICE-3D, CPU).

Plan fixé le 12 novembre 2026, avant ce calcul. Question : que fait le routeur d'une
photo de peau qui n'est pas une dermoscopie ? Limite annoncée des protocoles 1 et 2
(« pas de photos de peau prises au téléphone »).

- têtes des protocoles 1 et 2 (`heads/modality_router.pt`, `heads/modality_router_v2.pt`)
  inchangées, sans entraînement ni changement de seuil (0,9) ;
- images : les 6 294 recadrages de SLICE-3D Permissive déjà tirés pour l'étude de
  faisabilité (`scripts/slice3d_feasibility.py`, graine 0 : 294 malins, 6 000 bénins,
  quatre centres). Recadrages de photos du corps entier (Vectra WB360), pas de
  dermoscopie ; aucune n'a servi à ajuster le routeur (ajusté sur ODIR, ISIC 2019,
  Imagenette, MedMNIST) ;
- mêmes embeddings que le routeur (DINOv2 ViT-B/14, petit côté 256, recadrage 224), lus
  dans le cache de l'étude de faisabilité ; sa sélection est recalculée et doit
  retrouver exactement les clés du cache ;
- mesures, par tête : décisions (fond d'œil, dermoscopie, inconnu) sur l'ensemble, par
  centre et selon malin ou bénin ; taux envoyé en fond d'œil avec IC exact à 95 % ;
  probabilité de fond d'œil maximale ;
- critère fixé (sécurité) : borne haute de l'IC à 95 % du taux envoyé en fond d'œil
  au plus 0,5 %, pour chaque tête ;
- sans critère, lecture fixée à l'avance de la part « dermoscopie » contre « inconnu » :
  surtout « dermoscopie » → le routeur ne distingue pas une photo de peau d'une
  dermoscopie, et le modèle de peau (appris sur dermoscopie) recevrait ces photos : le
  type de photo doit rester déclaré ; surtout « inconnu » → le routeur refuse ces photos,
  ce qui est cohérent avec un modèle de peau appris sur dermoscopie, mais bloquerait un
  usage au téléphone.

Réserves : photos du corps entier recadrées (caméra 3D en cabinet), pas de vraies photos
de téléphone ; une seule source (ISIC 2024).

Exemple :
    python scripts/modality_router_skin_photo_stress.py --root data/raw/slice3d \\
        --checkpoint third_party/dinov2/dinov2_vitb14_timm.pth
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.modality_router_pathology_stress import HEADS, rates  # noqa: E402
from scripts.slice3d_feasibility import select  # noqa: E402
from scripts.train_modality_router import exact_ci  # noqa: E402
from src.eval.provenance import run_info  # noqa: E402
from src.models.embedding_cache import EmbeddingCache, file_fingerprint  # noqa: E402
from src.models.timm_backbones import DINOV2_IMAGE_SIZE  # noqa: E402
from src.pipeline.modality_classifier import MODALITIES, ModalityHead  # noqa: E402

SAFE_UPPER_BOUND = 0.005


def to_fundus(decided: np.ndarray) -> dict:
    """Envois en fond d'œil (erreur de sécurité) et critère : borne haute <= 0,5 %."""
    n, k = len(decided), int((decided == "fundus").sum())
    ci = exact_ci(k, n)
    return {"count": k, "n": n, "ci": ci, "threshold": SAFE_UPPER_BOUND, "met": ci[1] <= SAFE_UPPER_BOUND}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="DINOv2-B du routeur (empreinte vérifiée)")
    parser.add_argument("--cache", type=Path, default=Path("embeddings/slice3d_dinov2_vitb14.npz"))
    parser.add_argument("--out", type=Path, default=Path("results/modality_router_skin_photo_stress.json"))
    args = parser.parse_args()

    d = select(args.root)
    keys = [f"slice3d/{i}" for i in d["isic_id"]]
    if not args.cache.exists():
        raise FileNotFoundError(f"{args.cache} absent : lancer d'abord scripts/slice3d_feasibility.py")
    fingerprint = file_fingerprint(args.checkpoint)
    cache = EmbeddingCache(args.cache, meta={"checkpoint_sha256": fingerprint, "image_size": DINOV2_IMAGE_SIZE})
    if cache.missing(keys):
        raise ValueError(f"{len(cache.missing(keys))} images absentes du cache : sélection différente")
    X = cache.get(keys)
    site = d["attribution"].to_numpy()
    malignant = d["malignant"].astype(int).to_numpy()

    report = {
        "run": run_info(),
        "plan": "fixé le 12 novembre 2026 avant ce calcul (docstring du script)",
        "backbone": {"model": "dinov2_vitb14", "checkpoint_sha256": fingerprint},
        "images": {"total": len(d), "malignant": int(malignant.sum())},
        "heads": {},
    }
    for name, path in HEADS.items():
        head, _ = ModalityHead.load(path)
        probs = head.probabilities(X)
        decided = np.array(head.decide(probs))
        report["heads"][name] = {
            "all": rates(decided),
            "by_site": {s: rates(decided[site == s]) for s in sorted(set(site))},
            "malignant": {str(v): rates(decided[malignant == v]) for v in (0, 1)},
            "max_fundus_probability": float(probs[:, MODALITIES.index("fundus")].max()),
            "criterion_safe": to_fundus(decided),
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    for name, r in report["heads"].items():
        print(name, r["all"]["decisions"], "fond d'œil :", r["criterion_safe"])
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
