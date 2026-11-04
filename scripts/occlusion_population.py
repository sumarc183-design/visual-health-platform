#!/usr/bin/env python3
"""Occlusion sur un échantillon tiré au hasard, pour une ou plusieurs têtes.

scripts/occlusion_examples.py illustre sur 3 images par classe, choisies
parmi les positifs les mieux prédits : favorable par construction (audit du
4 novembre). Ici, pour chaque classe, `--positives` malades et
`--negatives` sains sont tirés **au hasard** dans la validation (graine
fixe), quelle que soit la prédiction : les erreurs sont donc comprises. Les
mêmes images passent dans chaque tête (ex. `none` puis `crop-mask`), ce qui
permet une comparaison appariée.

Mesures par image (src/explainability/occlusion.py) :
- part de l'effet d'occlusion hors du champ rétinien, à comparer à la part
  de l'image hors champ (au-dessus = le modèle s'appuie sur le fond plus
  que sa surface ne l'explique) ;
- variation du logit quand tout le hors-champ est remplacé par du gris.

Résumé par classe et par tête (médianes, quartiles, part des images où
l'effet hors champ dépasse la surface hors champ), et différence appariée
entre la première et la dernière tête (IC à 95 % par bootstrap sur les
images).

Exemple (GPU, ~10-15 min sur T4 pour ODIR) :
    python scripts/occlusion_population.py --dataset odir --data-root <ODIR> \\
        --checkpoint third_party/retfound/checkpoints/RETFound_mae_natureCFP.pth \\
        --heads heads/odir_none_head.pt heads/odir_crop-mask_head.pt \\
        --out results/occlusion_population_odir.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.gradcam_examples import load_val_dataset  # noqa: E402
from src.data.fundus_preprocess import common_mask  # noqa: E402
from src.explainability.gradcam import RETFoundClassifier  # noqa: E402
from src.explainability.occlusion import background_test, fundus_mask, occlusion_map, outside_share  # noqa: E402
from src.models.classifier_head import MultiLabelHead  # noqa: E402
from src.models.retfound_backbone import load_pretrained_backbone  # noqa: E402


def sample_indices(labels: np.ndarray, class_idx: int, n_pos: int, n_neg: int, rng) -> list[tuple[int, int]]:
    """(index, étiquette) tirés au hasard, sans regarder les prédictions."""
    picks = []
    for value, n in ((1, n_pos), (0, n_neg)):
        pool = np.flatnonzero(labels[:, class_idx] == value)
        chosen = rng.choice(pool, size=min(n, len(pool)), replace=False)
        picks += [(int(i), value) for i in chosen]
    return picks


def summarize(records: list[dict]) -> dict:
    def stats(values):
        values = np.array([v for v in values if v is not None], dtype=float)
        if not len(values):
            return None
        q1, median, q3 = np.percentile(values, [25, 50, 75])
        return {"median": float(median), "q1": float(q1), "q3": float(q3), "n": int(len(values))}

    excess = [
        r["part_effet_hors_champ"] > r["part_hors_champ"] for r in records if r["part_effet_hors_champ"] is not None
    ]
    return {
        "effect_outside": stats([r["part_effet_hors_champ"] for r in records]),
        "area_outside": stats([r["part_hors_champ"] for r in records]),
        "share_images_effect_above_area": float(np.mean(excess)) if excess else None,
        "gray_background_delta": stats([r["logit_fond_gris"] - r["logit"] for r in records]),
    }


def paired_median_diff(a: list, b: list, n_boot: int = 2000, seed: int = 0) -> dict | None:
    """Médiane de (b − a) sur les images où les deux mesures existent, IC 95 % par bootstrap."""
    pairs = np.array([(x, y) for x, y in zip(a, b, strict=True) if x is not None and y is not None])
    if len(pairs) < 5:
        return None
    diff = pairs[:, 1] - pairs[:, 0]
    rng = np.random.default_rng(seed)
    draws = [np.median(rng.choice(diff, len(diff))) for _ in range(n_boot)]
    return {"median_diff": float(np.median(diff)), "ci": [float(x) for x in np.percentile(draws, [2.5, 97.5])]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=["rfmid", "odir"], required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--heads", type=Path, nargs="+", required=True, help="Têtes à comparer (même jeu)")
    parser.add_argument("--classes", nargs="*", default=None, help="Défaut : toutes")
    parser.add_argument("--positives", type=int, default=15)
    parser.add_argument("--negatives", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--patch", type=int, default=32)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Demi-précision sur GPU (~2-3x plus rapide sur T4, chiffres légèrement approchés)",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    head_ckpts = [torch.load(h, map_location="cpu", weights_only=True) for h in args.heads]
    class_names = head_ckpts[0]["class_names"]
    if any(h["class_names"] != class_names or h["dataset"] != args.dataset for h in head_ckpts):
        parser.error("Les têtes doivent porter sur le même jeu et les mêmes classes.")
    _, labels = load_val_dataset(args.dataset, args.data_root, head_ckpts[0])
    classes = args.classes or list(class_names)
    rng = np.random.default_rng(args.seed)
    samples = {c: sample_indices(labels, class_names.index(c), args.positives, args.negatives, rng) for c in classes}

    vit_cache: dict[int, torch.nn.Module] = {}
    report = {
        "dataset": args.dataset,
        "seed": args.seed,
        "patch": args.patch,
        "stride": args.stride,
        "fp16": args.fp16,
        "heads": {},
    }
    for path, head_ckpt in zip(args.heads, head_ckpts, strict=True):
        preprocess, image_size = head_ckpt.get("preprocess", "none"), head_ckpt["image_size"]
        dataset, _ = load_val_dataset(args.dataset, args.data_root, head_ckpt)
        if image_size not in vit_cache:
            vit_cache[image_size] = load_pretrained_backbone(args.checkpoint, img_size=image_size, global_pool=True)
        head = MultiLabelHead(embedding_dim=1024, num_classes=len(class_names))
        head.load_state_dict(head_ckpt["state_dict"])
        model = RETFoundClassifier(vit_cache[image_size], head).to(args.device).eval()
        known_mask = torch.from_numpy(common_mask(image_size)) if preprocess == "crop-mask" else None

        records = []
        use_fp16 = args.fp16 and args.device.startswith("cuda")
        for class_name in classes:
            class_idx = class_names.index(class_name)
            print(f"[{preprocess}] {class_name} : {len(samples[class_name])} images...", flush=True)
            for idx, label in samples[class_name]:
                image_t, _ = dataset[idx]
                with torch.autocast("cuda", dtype=torch.float16, enabled=use_fp16):
                    heat, _ = occlusion_map(model, image_t, class_idx, args.patch, args.stride, args.batch_size)
                    mask = known_mask if known_mask is not None else fundus_mask(image_t)
                    bg = background_test(model, image_t, class_idx, mask)
                share = outside_share(heat, mask.numpy())
                records.append(
                    {
                        "classe": class_name,
                        "index": idx,
                        "label": label,
                        "p": float(torch.sigmoid(torch.tensor(bg["logit"]))),
                        **bg,
                        "part_effet_hors_champ": share if share == share else None,
                    }
                )
        summary = {}
        for class_name in classes:
            mine = [r for r in records if r["classe"] == class_name]
            summary[class_name] = {
                "positives": summarize([r for r in mine if r["label"] == 1]),
                "negatives": summarize([r for r in mine if r["label"] == 0]),
            }
        summary["toutes"] = {
            "positives": summarize([r for r in records if r["label"] == 1]),
            "negatives": summarize([r for r in records if r["label"] == 0]),
        }
        report["heads"][preprocess] = {"head": str(path), "summary": summary, "records": records}
        overall = summary["toutes"]["positives"]
        print(
            f"{preprocess} : effet hors champ médian {overall['effect_outside']['median']:.0%} "
            f"(surface hors champ {overall['area_outside']['median']:.0%}), effet > surface sur "
            f"{overall['share_images_effect_above_area']:.0%} des malades, fond gris "
            f"{overall['gray_background_delta']['median']:+.2f} (médiane)"
        )
        for class_name in classes:
            s = summary[class_name]["positives"]
            if s and s["effect_outside"]:
                print(
                    f"  {class_name:>5s} malades : hors champ {s['effect_outside']['median']:.0%} "
                    f"[{s['effect_outside']['q1']:.0%}-{s['effect_outside']['q3']:.0%}] pour "
                    f"{s['area_outside']['median']:.0%} de surface ; "
                    f"fond gris {s['gray_background_delta']['median']:+.2f}"
                )

    if len(args.heads) >= 2:
        first, last = list(report["heads"].values())[0], list(report["heads"].values())[-1]
        report["paired"] = {}
        for class_name in [*classes, "toutes"]:

            def pick(head, key, cls=class_name):
                return [
                    (r[key] if key == "part_effet_hors_champ" else r["logit_fond_gris"] - r["logit"])
                    for r in head["records"]
                    if r["label"] == 1 and (cls == "toutes" or r["classe"] == cls)
                ]

            report["paired"][class_name] = {
                "effect_outside": paired_median_diff(
                    pick(first, "part_effet_hors_champ"), pick(last, "part_effet_hors_champ")
                ),
                "gray_background_delta": paired_median_diff(pick(first, "delta"), pick(last, "delta")),
            }
        p = report["paired"]["toutes"]["effect_outside"]
        label = f"Apparié (malades, {list(report['heads'])[-1]} − {list(report['heads'])[0]}) : effet hors champ"
        if p is None:
            print(f"{label} non calculé (moins de 5 malades mesurés par les deux têtes)")
        else:
            print(f"{label} {p['median_diff']:+.0%} [{p['ci'][0]:+.0%} ; {p['ci'][1]:+.0%}]")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(f"Résultats : {args.out}")


if __name__ == "__main__":
    main()
