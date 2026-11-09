#!/usr/bin/env python3
"""Compare deux runs sur les mêmes images : différence d'AUROC B - A, par
classe et en macro, avec IC à 95 % par bootstrap apparié (mêmes tirages de
patients pour les deux runs).

Sert à décider entre deux variantes (ex. prétraitement none vs crop-mask)
sans se laisser tromper par le hasard : deux AUROC séparées ont chacune un
IC de ±0,03 à ±0,09 sur ODIR, mais leur différence appariée est bien plus
précise. Un écart est retenu seulement si son IC exclut 0.

À faire sur val : décider sur le test le rendrait optimiste.

Exemple :
    python scripts/compare_predictions.py results/pred_odir_none_val.npz \\
        results/pred_odir_crop-mask_val.npz --json results/compare_odir_val.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.metrics import paired_bootstrap_diff
from src.eval.predictions import validate_predictions  # noqa: E402
from src.eval.provenance import run_info  # noqa: E402


def load_predictions(path: Path) -> dict:
    with np.load(path) as data:
        return {name: data[name] for name in data.files}


def align(a: dict, b: dict) -> np.ndarray:
    """Probabilités de B remises dans l'ordre des images de A, après
    vérification que ce sont les mêmes images, étiquettes et classes."""
    for name, pred in (("A", a), ("B", b)):
        try:
            validate_predictions(pred, f"run {name}")
        except ValueError as error:
            raise SystemExit(str(error)) from None
    if list(a["class_names"]) != list(b["class_names"]):
        raise SystemExit("Les deux runs n'ont pas les mêmes classes.")
    if set(a["keys"].tolist()) != set(b["keys"].tolist()):
        raise SystemExit("Les deux runs n'ont pas été évalués sur les mêmes images (même jeu, même split ?).")
    position = {key: i for i, key in enumerate(b["keys"].tolist())}
    order = np.array([position[key] for key in a["keys"].tolist()])
    if not np.array_equal(a["y_true"], b["y_true"][order]):
        raise SystemExit("Mêmes images mais étiquettes différentes (--odir-labels différent ?).")
    return b["probs"][order]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_a", type=Path, help="Prédictions de référence (.npz de run_baseline.py --predictions)")
    parser.add_argument("run_b", type=Path, help="Prédictions de la variante comparée")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    a, b = load_predictions(args.run_a), load_predictions(args.run_b)
    prob_b = align(a, b)
    class_names = a["class_names"].tolist()
    diff = paired_bootstrap_diff(
        a["y_true"], a["probs"], prob_b, class_names, a["groups"].tolist(), n_boot=args.bootstrap
    )

    print(f"AUROC B - A   (A = {args.run_a.name}, B = {args.run_b.name}, {len(a['keys'])} images)")
    for name, d in diff.items():
        lo, hi = d["ci"]
        verdict = "B meilleur" if lo > 0 else "A meilleur" if hi < 0 else "pas de différence établie"
        n_pos = "" if name == "macro" else f"  (n+ = {int(a['y_true'][:, class_names.index(name)].sum())})"
        print(f"  {name:>8s} : {d['diff']:+.3f}  [IC 95 % {lo:+.3f} ; {hi:+.3f}]  {verdict}{n_pos}")
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps({"run": run_info(), "a": str(args.run_a), "b": str(args.run_b), **diff}, indent=2)
        )
        print(f"Comparaison JSON : {args.json}")


if __name__ == "__main__":
    main()
