#!/usr/bin/env python3
"""Les modèles regardent-ils les lésions ? Mesures contre les masques d'experts
de la segmentation IDRiD (CPU, à partir de scripts/localization_maps.py).

Plan fixé avant de lire les résultats, pour RETFound et RETFound-Green :

1. **Occlusion (81 images)**, tête RFMiD servie par predict.py, jamais
   entraînée sur IDRiD. Pour chaque image, AUROC au pixel de la carte contre
   les lésions (union MA, HE, EX, SE), puis par type, et enrichissement dans
   les 5 % de pixels les plus importants. Référence : la carte moyenne des
   **autres** images, car lésions et cartes ont des positions typiques (une
   AUROC > 0,5 peut venir de la seule position). Le disque optique, structure
   normale, sert de contrôle : il ne fait pas partie de la rétinopathie.
2. **Segmentation (54 images d'entraînement, 27 de test, lues une fois)**.
   Régression logistique par case, sur les tokens de patch standardisés
   (C = 1), sans réglage, par type de lésion et disque optique. Mesure :
   précision moyenne (AP) sur une grille commune de 56×56, sur la rétine.
   Référence : fréquence de la lésion à chaque position dans l'entraînement,
   qui ne regarde pas l'image.

IC à 95 % par bootstrap sur les images, différences appariées.

Exemple :
    python scripts/localization_analysis.py --data-root <IDRiD> --maps-dir results/localization \\
        --out results/localization_idrid.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.fundus_preprocess import common_mask  # noqa: E402
from src.data.idrid_dataset import LESION_DIRS, LESIONS, IDRiDSegmentation  # noqa: E402
from src.explainability.localization import cell_targets, map_auroc, resize_map, top_enrichment  # noqa: E402

SIZE = 224
EVAL_GRID = 56
FILES = {"RETFound": "retfound", "RETFound-Green": "green"}
N_BOOT = 1000


def load_masks(root: Path) -> dict[str, dict[str, np.ndarray]]:
    """Identifiant -> lésion -> masque booléen (224, 224)."""
    masks = {}
    for split in ("train", "test"):
        dataset = IDRiDSegmentation(root, split=split, image_size=SIZE)
        for i, image_id in enumerate(dataset.ids):
            masks[image_id] = {k: v > 0 for k, v in dataset.masks(i).items()}
    return masks


def mean_ci(values: np.ndarray, rng: np.random.Generator) -> dict:
    """Moyenne et IC à 95 % par bootstrap sur les images (NaN ignorés)."""
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return {"mean": None, "ci": None, "images": 0}
    draws = rng.choice(values, (N_BOOT, len(values))).mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci": [float(x) for x in np.percentile(draws, [2.5, 97.5])],
        "images": len(values),
    }


def occlusion_scores(maps: np.ndarray, masks: list[dict], key: str, retina: np.ndarray) -> dict[str, np.ndarray]:
    """Par image : AUROC de sa carte, AUROC de la carte moyenne des autres, enrichissement."""
    total = maps.sum(axis=0)
    own, other, enrichment = [], [], []
    for i, mask in enumerate(masks):
        target = mask[key] if key != "lesions" else np.logical_or.reduce([mask[k] for k in LESIONS])
        others = (total - maps[i]) / (len(maps) - 1)
        own.append(map_auroc(maps[i], target, retina))
        other.append(map_auroc(others, target, retina))
        enrichment.append(top_enrichment(maps[i], target, retina))
    return {"own": np.array(own), "other": np.array(other), "enrichment": np.array(enrichment)}


def occlusion_report(maps_by_model: dict, logits_by_model: dict, masks: list[dict], seed: int) -> dict:
    retina = common_mask(SIZE)
    rng = np.random.default_rng(seed)
    report = {}
    scores = {}
    for model, maps in maps_by_model.items():
        resized = np.stack([resize_map(m, SIZE) for m in maps])
        report[model] = {"logit_median": float(np.median(logits_by_model[model]))}
        for key in ("lesions", *LESION_DIRS):
            s = occlusion_scores(resized, masks, key, retina)
            scores[model, key] = s
            report[model][key] = {
                "auroc": mean_ci(s["own"], rng),
                "auroc_other_images": mean_ci(s["other"], rng),
                "auroc_minus_other_images": mean_ci(s["own"] - s["other"], rng),
                "top5pct_enrichment": mean_ci(s["enrichment"], rng),
            }
    if len(maps_by_model) == 2:
        a, b = list(maps_by_model)
        report["green_minus_retfound"] = {
            key: mean_ci(scores[b, key]["own"] - scores[a, key]["own"], rng) for key in ("lesions", *LESION_DIRS)
        }
    return report


def upsample(grid_values: np.ndarray) -> np.ndarray:
    """Grille (G, G) -> grille d'évaluation (EVAL_GRID, EVAL_GRID), sans interpolation."""
    return np.kron(grid_values, np.ones((EVAL_GRID // len(grid_values),) * 2))


def segmentation_scores(tokens: np.ndarray, targets: list[np.ndarray], train: np.ndarray, retina: np.ndarray):
    """Probabilités par case, entraînées sur `train`, ramenées à la grille d'évaluation."""
    grid = int(np.sqrt(tokens.shape[1]))
    cells = [cell_targets(t, grid, retina) for t in targets]
    inside = cells[0][1].ravel()
    X_train = tokens[train][:, inside].reshape(-1, tokens.shape[-1])
    y_train = np.concatenate([cells[i][0].ravel()[inside] for i in np.flatnonzero(train)])
    scaler = StandardScaler().fit(X_train)
    model = LogisticRegression(C=1.0, max_iter=1000).fit(scaler.transform(X_train), y_train)
    out = []
    for image_tokens in tokens:
        probs = model.predict_proba(scaler.transform(image_tokens.reshape(-1, tokens.shape[-1])))[:, 1]
        out.append(upsample(probs.reshape(grid, grid)))
    return np.stack(out)


def paired_ap(y: np.ndarray, a: np.ndarray, b: np.ndarray | None, rng: np.random.Generator) -> dict:
    """AP de `a` (ou a − b), y et scores en (images, cases). IC par images."""

    def score(idx: np.ndarray, values: np.ndarray) -> float:
        return average_precision_score(y[idx].ravel(), values[idx].ravel())

    full = np.arange(len(y))
    point = score(full, a) - (score(full, b) if b is not None else 0.0)
    draws = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].sum() == 0:
            continue
        draws.append(score(idx, a) - (score(idx, b) if b is not None else 0.0))
    return {"value": float(point), "ci": [float(x) for x in np.percentile(draws, [2.5, 97.5])]}


def segmentation_report(tokens_by_model: dict, ids: list[str], splits: np.ndarray, masks: dict, seed: int) -> dict:
    retina = common_mask(SIZE)
    train = splits == "train"
    rng = np.random.default_rng(seed)
    report = {}
    for key in LESION_DIRS:
        targets = [masks[i][key] for i in ids]
        eval_y, inside = zip(*[cell_targets(t, EVAL_GRID, retina) for t in targets], strict=True)
        eval_y, inside = np.stack(eval_y), inside[0]
        prior = upsample(np.zeros((EVAL_GRID, EVAL_GRID))) + eval_y[train].mean(axis=0)
        y_test = eval_y[~train][:, inside]
        prior_test = np.repeat(prior[inside][None], len(y_test), axis=0)
        entry = {
            "test_images_with_lesion": int(y_test.any(axis=1).sum()),
            "prevalence_cells_test": float(y_test.mean()),
            "ap_position_only": paired_ap(y_test, prior_test, None, rng),
        }
        predictions = {}
        for model, tokens in tokens_by_model.items():
            probs = segmentation_scores(tokens, targets, train, retina)[~train][:, inside]
            predictions[model] = probs
            entry[model] = {
                "ap": paired_ap(y_test, probs, None, rng),
                "ap_minus_position": paired_ap(y_test, probs, prior_test, rng),
            }
        if len(predictions) == 2:
            a, b = list(predictions)
            entry["green_minus_retfound"] = paired_ap(y_test, predictions[b], predictions[a], rng)
        report[key] = entry
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--maps-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    masks = load_masks(args.data_root)
    data = {m: np.load(args.maps_dir / f"localization_{f}.npz") for m, f in FILES.items()}
    first = data["RETFound"]
    ids, splits = [str(i) for i in first["ids"]], first["splits"]
    for model, d in data.items():
        if [str(i) for i in d["ids"]] != ids:
            raise ValueError(f"{model} : images différentes de RETFound")
    image_masks = [masks[i] for i in ids]
    report = {
        "images": len(ids),
        "occlusion": occlusion_report(
            {m: d["maps"] for m, d in data.items()}, {m: d["logits"] for m, d in data.items()}, image_masks, args.seed
        ),
        "segmentation": segmentation_report({m: d["tokens"] for m, d in data.items()}, ids, splits, masks, args.seed),
    }

    print(f"Occlusion, {len(ids)} images : AUROC [IC] | carte des autres images | écart | enrichissement top 5 %")
    for model in FILES:
        r = report["occlusion"][model]
        print(f"{model} (logit médian {r['logit_median']:+.2f})")
        for key in ("lesions", *LESION_DIRS):
            e = r[key]
            d = e["auroc_minus_other_images"]
            print(
                f"  {key:8s} {e['auroc']['mean']:.3f} [{e['auroc']['ci'][0]:.3f} ; {e['auroc']['ci'][1]:.3f}] | "
                f"{e['auroc_other_images']['mean']:.3f} | {d['mean']:+.3f} [{d['ci'][0]:+.3f} ; {d['ci'][1]:+.3f}] | "
                f"x{e['top5pct_enrichment']['mean']:.2f}"
            )
    for key, d in report["occlusion"].get("green_minus_retfound", {}).items():
        print(f"  Green − RETFound {key:8s} {d['mean']:+.3f} [{d['ci'][0]:+.3f} ; {d['ci'][1]:+.3f}]")
    print("Segmentation, test : AP [IC] | position seule | écart à la position")
    for key, e in report["segmentation"].items():
        print(
            f"{key} (cases positives {e['prevalence_cells_test']:.1%}, position seule AP "
            f"{e['ap_position_only']['value']:.3f})"
        )
        for model in FILES:
            a, d = e[model]["ap"], e[model]["ap_minus_position"]
            print(
                f"  {model:15s} {a['value']:.3f} [{a['ci'][0]:.3f} ; {a['ci'][1]:.3f}] | "
                f"{d['value']:+.3f} [{d['ci'][0]:+.3f} ; {d['ci'][1]:+.3f}]"
            )
        d = e["green_minus_retfound"]
        print(f"  Green − RETFound {d['value']:+.3f} [{d['ci'][0]:+.3f} ; {d['ci'][1]:+.3f}]")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(f"Résultats : {args.out}")


if __name__ == "__main__":
    main()
