#!/usr/bin/env python3
"""Nouvelle confirmation sur des bases vues par aucun des deux modèles : Messidor-2 et IDRiD (CPU).

Plan fixé le 13 novembre 2026, avant tout calcul d'embedding sur ces bases. Seuls la
structure des fichiers et le décompte des stades ont été lus (Messidor-2 : 1 744 images
gradables, stades 0 à 4 : 1 017 / 270 / 347 / 75 / 35 ; IDRiD : 516 images, 168 / 25 /
168 / 93 / 62).

Pourquoi : DDR fait partie du pré-entraînement de RETFound-Green (article, tableau S1 :
AIROGS, DDR, ODIR), pas de RETFound ; la confirmation du 12 novembre et les mesures du
dépistage sur DDR ne sont donc pas indépendantes. Messidor-2 (France) et IDRiD (Inde)
n'apparaissent dans aucun des deux pré-entraînements (RETFound : MEH-MIDAS et EyePACS)
ni nulle part dans le projet pour la rétinopathie.

Données :
- Messidor-2 (principal) : images ADCIS, stades adjudiqués de Krause et al. 2018
  (Kaggle `google-brain/messidor2-dr-grades`, CC0) ; 4 images non gradables écartées ;
  regroupement par examen (deux yeux) si la liste des paires est fournie avec les images,
  sinon par image (réserve) ;
- IDRiD, partie « B. Disease Grading » (secondaire) : 516 images, apprentissage et test
  officiels réunis (rien n'y est appris), regroupement par image.

Embeddings (CPU, poids locaux ; RETFound vérifié identique au cache RFMiD, cosinus
minimal 0,9999998 sur 20 images) : `crop-mask`, RETFound 224 px, Green 392 px.

Mesure 1, modèle par défaut (même méthode que DDR) : tête « rétinopathie » (colonne DR)
apprise sur l'entraînement RFMiD pour chaque modèle (réglages de la baseline, graine
42). Critère : écart d'AUROC « toute rétinopathie » (stade >= 1), Green − RETFound,
bootstrap apparié sur les groupes (1 000 tirages), borne basse > 0 sur Messidor-2.
IDRiD : même calcul, rapporté comme seconde confirmation (critère identique, lu à part).

Mesure 2, dépistage (`predict.py --dr-screening`, tête et seuil servis, inchangés) :
sensibilité « toute rétinopathie », IC bootstrap sur les groupes ; critère (le même que
sur DDR) : borne basse >= 0,80 sur Messidor-2. Rapportés : spécificité, VPP, sensibilité
« à adresser » (stade >= 2), même calcul sur IDRiD.

Secondaires, sans critère : AUROC de chaque modèle, « à adresser » ; sensibilité et
spécificité de la tête RFMiD Green au seuil de validation RFMiD.

Conséquences prévues : mesure 1 confirmée → le choix de Green par défaut repose sur une
confirmation indépendante ; sinon → à rediscuter avec le tuteur. Mesure 2 atteinte → le
réglage dépistage est documenté avec les chiffres Messidor-2 ; sinon → ses chiffres DDR
sont retirés et le réglage est signalé comme non validé.

Réserves : Messidor-2 ancien (2004-2010) et d'un seul pays ; IDRiD petit (IC larges) ;
licences : Messidor-2 usage recherche, IDRiD CC BY 4.0.

Exemple :
    python scripts/confirmation_external.py --dataset idrid --root data/raw/idrid_grading \\
        --rfmid-root data/raw/rfmid --embeddings embeddings
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.dr_screening_sensitivity import rate_ci  # noqa: E402
from scripts.external_validation import quiet_scores  # noqa: E402
from scripts.green_analysis import cached  # noqa: E402
from scripts.pad_ufes20_skin import auroc_ci  # noqa: E402
from scripts.robustness_checks import load  # noqa: E402
from src.data.fundus_preprocess import load_fundus_image  # noqa: E402
from src.eval.decision import choose_threshold, decision_metrics  # noqa: E402
from src.eval.metrics import paired_bootstrap_diff  # noqa: E402
from src.eval.provenance import run_info  # noqa: E402
from src.models.classifier_head import MultiLabelHead  # noqa: E402
from src.models.embedding_cache import EmbeddingCache, file_fingerprint  # noqa: E402
from src.models.retfound_backbone import IMAGENET_MEAN, IMAGENET_STD, RETFoundBackbone  # noqa: E402
from src.models.retfound_green import GREEN_IMAGE_SIZE, RETFoundGreenBackbone  # noqa: E402

CHECKPOINTS = {
    "retfound": Path("third_party/retfound/checkpoints/RETFound_mae_natureCFP.pth"),
    "green": Path("third_party/retfound_green/retfoundgreen_statedict.pth"),
}
SIZES = {"retfound": 224, "green": GREEN_IMAGE_SIZE}
CACHE_MODEL = {"retfound": "RETFound", "green": "RETFound-Green"}
SCREENING_HEAD = Path("heads/dr_screening_green_head.pt")
SCREENING_THRESHOLDS = Path("results/decision_dr_screening.json")
CRITERION_SENSITIVITY = 0.80


def idrid_table(root: Path) -> pd.DataFrame:
    """Clé, chemin, stade et groupe (image) des 516 images de gradation IDRiD."""
    base = root / "B. Disease Grading"
    frames = []
    for labels, images in (
        ("a. IDRiD_Disease Grading_Training Labels.csv", "a. Training Set"),
        ("b. IDRiD_Disease Grading_Testing Labels.csv", "b. Testing Set"),
    ):
        df = pd.read_csv(base / "2. Groundtruths" / labels).iloc[:, :2]
        df.columns = ["name", "grade"]
        df["path"] = [base / "1. Original Images" / images / f"{n}.jpg" for n in df["name"]]
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["key"] = "idrid/" + df["name"]
    df["group"] = df["key"]
    return df[["key", "path", "grade", "group"]]


def messidor_table(root: Path) -> pd.DataFrame:
    """Images gradables de Messidor-2 ; groupe = examen si une liste de paires est fournie."""
    grades = pd.read_csv(root / "messidor_data.csv")
    grades = grades[grades["adjudicated_gradable"] == 1].reset_index(drop=True)
    paths = {p.name: p for p in root.rglob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif")}
    missing = set(grades["image_id"]) - set(paths)
    if missing:
        raise FileNotFoundError(f"{len(missing)} images Messidor-2 absentes, ex. {sorted(missing)[:3]}")
    df = pd.DataFrame(
        {
            "key": "messidor2/" + grades["image_id"],
            "path": [paths[i] for i in grades["image_id"]],
            "grade": grades["adjudicated_dr_grade"].astype(int),
        }
    )
    pairs = sorted(root.rglob("*pairs*.csv"))
    if pairs:
        exam = pd.read_csv(pairs[0])  # colonnes attendues : image_id, exam_id
        df["group"] = df["key"].map(dict(zip("messidor2/" + exam.iloc[:, 0], exam.iloc[:, 1], strict=True)))
    else:
        df["group"] = df["key"]
    return df


def backbone(model: str):
    if model == "green":
        return RETFoundGreenBackbone(CHECKPOINTS["green"]).eval()
    return RETFoundBackbone(CHECKPOINTS["retfound"], img_size=224, embedding_dim=1024, trust_checkpoint=True).eval()


@torch.no_grad()
def embeddings(table: pd.DataFrame, model: str, cache_path: Path) -> np.ndarray:
    meta = {
        "checkpoint_sha256": file_fingerprint(CHECKPOINTS[model]),
        "image_size": SIZES[model],
        "preprocess": "crop-mask",
    }
    cache = EmbeddingCache(cache_path, meta=meta)
    keys = list(table["key"])
    todo = set(cache.missing(keys))
    if todo:
        net = backbone(model)
        mean = getattr(net, "input_mean", IMAGENET_MEAN)
        std = getattr(net, "input_std", IMAGENET_STD)
        rows = table[table["key"].isin(todo)]
        start = time.time()
        for b in range(0, len(rows), 16):
            part = rows.iloc[b : b + 16]
            x = torch.stack([load_fundus_image(p, SIZES[model], "crop-mask") for p in part["path"]])
            cache.add(list(part["key"]), net((x - mean) / std).numpy())
            print(f"  {model} {b + len(part)}/{len(rows)} en {time.time() - start:.0f} s")
        cache.save()
    return cache.get(keys)


def screening(y: np.ndarray, p: np.ndarray, threshold: float, groups: list[str], grade: np.ndarray) -> dict:
    decided = p >= threshold
    return {
        **decision_metrics(y, p, threshold),
        "sensitivity_ci": rate_ci(y, decided, groups, True),
        "specificity_ci": rate_ci(y, decided, groups, False),
        "referable_sensitivity": float(decided[grade >= 2].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=["messidor2", "idrid"], required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--rfmid-root", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, default=Path("embeddings"))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out = args.out or Path(f"results/confirmation_{args.dataset}.json")

    table = idrid_table(args.root) if args.dataset == "idrid" else messidor_table(args.root)
    grade = table["grade"].to_numpy()
    y = (grade >= 1).astype(np.float32)
    groups = list(table["group"].astype(str))

    train, val = load("rfmid", args.rfmid_root, "train"), load("rfmid", args.rfmid_root, "val")
    report = {
        "run": run_info(),
        "plan": "fixé le 13 novembre 2026 avant tout calcul d'embedding (docstring du script)",
        "dataset": args.dataset,
        "images": len(table),
        "groups": len(set(groups)),
        "by_grade": {int(g): int((grade == g).sum()) for g in np.unique(grade)},
        "models": {},
    }
    transfer = {}
    for model in ("retfound", "green"):
        X = embeddings(table, model, args.embeddings / f"{args.dataset}_crop-mask_{SIZES[model]}_{model}.npz")
        X_tr = cached(args.embeddings, CACHE_MODEL[model], "rfmid", train.image_keys)
        X_val = cached(args.embeddings, CACHE_MODEL[model], "rfmid", val.image_keys)
        y_tr = train.df["DR"].to_numpy(np.float32)
        p = quiet_scores(X_tr, y_tr[:, None], np.vstack([X_val, X]))
        p_val, p_ext = p[: len(X_val)], p[len(X_val) :]
        threshold = choose_threshold(val.df["DR"].to_numpy(np.float32), p_val, "sensitivity", 0.90)
        transfer[model] = p_ext
        report["models"][model] = {
            "auroc_any_dr": auroc_ci(y, p_ext, groups),
            "auroc_referable": auroc_ci((grade >= 2).astype(np.float32), p_ext, groups),
            "rfmid_threshold": screening(y, p_ext, threshold, groups, grade),
        }
        if model == "green":
            X_green = X
    diff = paired_bootstrap_diff(y[:, None], transfer["retfound"][:, None], transfer["green"][:, None], ["DR"], groups)
    report["green_minus_retfound"] = diff["DR"]

    head_ckpt = torch.load(SCREENING_HEAD, map_location="cpu", weights_only=True)
    head = MultiLabelHead(X_green.shape[1], 1)
    head.load_state_dict(head_ckpt["state_dict"])
    with torch.no_grad():
        p_screen = torch.sigmoid(head(torch.from_numpy(X_green))).numpy()[:, 0]
    threshold = json.loads(SCREENING_THRESHOLDS.read_text())["classes"]["DR"]["threshold_sensitivity"]["threshold"]
    report["dr_screening"] = screening(y, p_screen, threshold, groups, grade)

    report["criteria"] = {
        "default_model": {"diff": diff["DR"], "confirmed": bool(diff["DR"]["ci"][0] > 0)},
        "screening": {
            "sensitivity": report["dr_screening"]["sensitivity"],
            "lower_bound": report["dr_screening"]["sensitivity_ci"][0],
            "threshold": CRITERION_SENSITIVITY,
            "met": bool(report["dr_screening"]["sensitivity_ci"][0] >= CRITERION_SENSITIVITY),
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print({k: report[k] for k in ("dataset", "images", "groups", "by_grade")})
    for model, m in report["models"].items():
        print(f"{model:8s} AUROC toute RD {m['auroc_any_dr']}, à adresser {m['auroc_referable']['auroc']:.3f}")
    print("Green − RETFound :", diff["DR"])
    s = report["dr_screening"]
    print(
        f"Dépistage : sensibilité {s['sensitivity']:.3f} {s['sensitivity_ci']}, spécificité {s['specificity']:.3f}, "
        f"VPP {s['ppv']:.3f}, à adresser {s['referable_sensitivity']:.3f}"
    )
    print("Critères :", report["criteria"])
    print(f"-> {out}")


if __name__ == "__main__":
    main()
