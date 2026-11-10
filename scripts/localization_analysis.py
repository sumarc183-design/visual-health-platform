#!/usr/bin/env python3
"""Les modèles regardent-ils les lésions ? Mesures contre les masques d'experts
de la segmentation IDRiD (CPU, à partir de scripts/localization_maps.py).

Plan fixé avant de lire les résultats, pour RETFound et RETFound-Green :

1. **Occlusion (81 images)**, tête RFMiD servie par predict.py, jamais
   entraînée sur IDRiD. Pour chaque image, AUROC au pixel de la carte contre
   les lésions (union MA, HE, EX, SE), puis par type, et enrichissement dans
   exactement 5 % des pixels, ceux de plus fort score (et 2 % et 10 %, en stabilité). Référence : la carte moyenne des
   **autres** images, car lésions et cartes ont des positions typiques (une
   AUROC > 0,5 peut venir de la seule position). Le disque optique, structure
   normale, sert de contrôle : il ne fait pas partie de la rétinopathie.
2. **Segmentation (54 images d'entraînement, 27 de test, lues une fois)**.
   Régression logistique par case, sur les tokens de patch standardisés
   (C = 1), sans réglage, par type de lésion et disque optique. Mesure :
   précision moyenne (AP) sur une grille commune de 56×56, sur la rétine.
   Référence : fréquence de la lésion à chaque position dans l'entraînement,
   qui ne regarde pas l'image.

IC à 95 % par bootstrap sur les images, différences appariées. Cibles et
probabilités du test archivées (`localization_predictions.npz`) : les AP se
recalculent sans réentraîner.

Exploratoire, ajouté après lecture des résultats : Green à la grille de
RETFound (tokens moyennés en 14×14), car sa grille native de 28×28 l'avantage ;
l'entrée reste en 392 px. Planche d'exemples tirés au hasard dans le test
(`--figure`).

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
from src.data.idrid_dataset import (  # noqa: E402
    LESION_DIRS,
    LESIONS,
    IDRiDSegmentation,
    check_official_split,
    compare_inventories,
    input_inventory,
)
from src.eval.provenance import run_info  # noqa: E402
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


def lesion_target(mask: dict, key: str) -> np.ndarray:
    return mask[key] if key != "lesions" else np.logical_or.reduce([mask[k] for k in LESIONS])


def occlusion_scores(maps: np.ndarray, masks: list[dict], key: str, retina: np.ndarray) -> dict[str, np.ndarray]:
    """Par image : AUROC de sa carte, AUROC de la carte moyenne des autres, enrichissement."""
    total = maps.sum(axis=0)
    own, other, enrichment = [], [], []
    for i, mask in enumerate(masks):
        target = lesion_target(mask, key)
        others = (total - maps[i]) / (len(maps) - 1)
        own.append(map_auroc(maps[i], target, retina))
        other.append(map_auroc(others, target, retina))
        enrichment.append(top_enrichment(maps[i], target, retina))
    return {"own": np.array(own), "other": np.array(other), "enrichment": np.array(enrichment)}


def fraction_enrichment(maps: np.ndarray, masks: list[dict], key: str, retina: np.ndarray, fraction: float):
    return np.array(
        [top_enrichment(m, lesion_target(mask, key), retina, fraction) for m, mask in zip(maps, masks, strict=True)]
    )


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
                "top_fraction_enrichment": {
                    str(f): mean_ci(fraction_enrichment(resized, masks, key, retina, f), rng) for f in (0.02, 0.10)
                },
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


def pool_tokens(tokens: np.ndarray) -> np.ndarray:
    """Tokens (N, 28², D) -> (N, 14², D), moyennés par blocs de 2×2."""
    n, count, dim = tokens.shape
    grid = int(np.sqrt(count))
    blocks = tokens.reshape(n, grid // 2, 2, grid // 2, 2, dim)
    return blocks.mean(axis=(2, 4)).reshape(n, -1, dim)


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


def segmentation_report(
    tokens_by_model: dict, ids: list[str], splits: np.ndarray, masks: dict, seed: int, archive: dict | None = None
) -> dict:
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
        if archive is not None:
            archive[f"{key}_y"] = y_test
            archive[f"{key}_position_only"] = prior_test[0].astype(np.float32)
        for model, tokens in tokens_by_model.items():
            probs = segmentation_scores(tokens, targets, train, retina)[~train][:, inside]
            predictions[model] = probs
            if archive is not None:
                archive[f"{key}_{FILES[model]}"] = probs.astype(np.float32)
            entry[model] = {
                "ap": paired_ap(y_test, probs, None, rng),
                "ap_minus_position": paired_ap(y_test, probs, prior_test, rng),
            }
        if len(predictions) == 2:
            a, b = list(predictions)
            entry["green_minus_retfound"] = paired_ap(y_test, predictions[b], predictions[a], rng)
            pooled = segmentation_scores(pool_tokens(tokens_by_model[b]), targets, train, retina)[~train][:, inside]
            entry["exploratory_green_grid_14"] = {
                "ap": paired_ap(y_test, pooled, None, rng),
                "minus_retfound": paired_ap(y_test, pooled, predictions[a], rng),
            }
        report[key] = entry
    return report


def save_figure(path: Path, root: Path, data: dict, masks: dict, seed: int, count: int = 4) -> None:
    """Images de test tirées au hasard : image, masques d'experts, cartes des deux modèles."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dataset = IDRiDSegmentation(root, split="test", image_size=SIZE)
    picks = np.random.default_rng(seed).choice(len(dataset), count, replace=False)
    colors = {"OD": (0, 1, 1), "EX": (1, 1, 0), "SE": (1, 0, 1), "HE": (1, 0, 0), "MA": (0, 1, 0)}
    retina = common_mask(SIZE)
    fig, axes = plt.subplots(count, 4, figsize=(12, 3 * count), squeeze=False)
    for row, i in zip(axes, picks, strict=True):
        image_id = dataset.ids[i]
        image = dataset.image(int(i)).permute(1, 2, 0).numpy()
        overlay = image.copy()
        for key, color in colors.items():
            overlay[masks[image_id][key]] = color
        row[0].imshow(image)
        row[0].set_title(image_id, fontsize=9)
        row[1].imshow(overlay)
        row[1].set_title("Experts : disque (cyan), lésions", fontsize=9)
        for ax, (model, d) in zip(row[2:], data.items(), strict=True):
            k = [str(x) for x in d["ids"]].index(image_id)
            heat = np.where(retina, resize_map(d["maps"][k], SIZE), np.nan)
            ax.imshow(image)
            ax.imshow(heat, cmap="magma", alpha=0.65, vmin=0, vmax=max(1e-6, float(np.nanpercentile(heat, 99))))
            ax.set_title(f"{model}, occlusion", fontsize=9)
        for ax in row:
            ax.axis("off")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def check_inputs(archive, data_root: Path) -> dict:
    """Compare l'inventaire du manifeste (images et masques lus pour les cartes) aux
    fichiers présents. Archive sans inventaire : signalée non vérifiée, pas refusée."""
    if "manifest" not in archive.files:
        return {"verified": False, "reason": "archive sans manifeste (antérieure au 10 novembre 2026)"}
    recorded = json.loads(str(archive["manifest"])).get("inputs")
    if recorded is None:
        return {"verified": False, "reason": "manifeste sans inventaire des entrées"}
    ids_by_split: dict[str, list[str]] = {}
    for image_id, split in zip(archive["ids"], archive["splits"], strict=True):
        ids_by_split.setdefault(str(split), []).append(str(image_id))
    differing = compare_inventories(recorded, input_inventory(data_root, ids_by_split))
    if differing:
        raise ValueError(
            f"entrées IDRiD différentes de celles des cartes : {differing[:5]} ({len(differing)} fichiers)"
        )
    return {"verified": True, "digest": recorded["digest"], "absent_masks": len(recorded["absent_masks"])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--maps-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--figure", type=Path, default=None, help="Planche d'exemples tirés au hasard (PNG)")
    args = parser.parse_args()

    masks = load_masks(args.data_root)
    data = {m: np.load(args.maps_dir / f"localization_{f}.npz") for m, f in FILES.items()}
    first = data["RETFound"]
    ids, splits = [str(i) for i in first["ids"]], first["splits"]
    for model, d in data.items():
        if [str(i) for i in d["ids"]] != ids:
            raise ValueError(f"{model} : images différentes de RETFound")
    if [str(s) for s in data["RETFound-Green"]["splits"]] != [str(s) for s in splits]:
        raise ValueError("RETFound-Green : découpage différent de RETFound")
    check_official_split(ids, [str(s) for s in splits])
    image_masks = [masks[i] for i in ids]
    archive: dict = {"ids_test": np.array(ids)[splits == "test"]}
    report = {
        "run": run_info(),
        "maps_manifest": {m: str(d["manifest"]) if "manifest" in d.files else None for m, d in data.items()},
        "inputs_check": {m: check_inputs(d, args.data_root) for m, d in data.items()},
        "images": len(ids),
        "occlusion": occlusion_report(
            {m: d["maps"] for m, d in data.items()}, {m: d["logits"] for m, d in data.items()}, image_masks, args.seed
        ),
        "segmentation": segmentation_report(
            {m: d["tokens"] for m, d in data.items()}, ids, splits, masks, args.seed, archive
        ),
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
        x = e["exploratory_green_grid_14"]
        d = x["minus_retfound"]
        print(
            f"  Green grille 14×14 {x['ap']['value']:.3f} | "
            f"− RETFound {d['value']:+.3f} [{d['ci'][0]:+.3f} ; {d['ci'][1]:+.3f}]"
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    np.savez_compressed(args.out.parent / "localization_predictions.npz", **archive)
    print(f"Résultats : {args.out}")
    if args.figure is not None:
        save_figure(args.figure, args.data_root, data, masks, args.seed)
        print(f"Planche : {args.figure}")


if __name__ == "__main__":
    main()
