#!/usr/bin/env python3
"""Fine-tuning partiel de RETFound sur RFMiD (GPU), configuration fixée à l'avance.

Configuration préspécifiée (pas de recherche d'hyperparamètres : un seul
réglage, comparé à la tête linéaire figée par différence appariée) :
- 4 derniers blocs du ViT-L + tête entraînés, le reste figé ;
- AdamW, lr 2e-5 (blocs) et 1e-3 (tête), weight decay 0,05, warmup d'une
  epoch puis décroissance cosinus, mini-batchs de 32, demi-précision ;
- augmentations : retournement, rotation ±15°, luminosité, contraste,
  saturation, masque commun réappliqué (src/models/finetune.py) ;
- 30 epochs au plus ; on garde l'epoch à la meilleure AUROC macro de
  validation (arrêt après 5 epochs sans progrès).

Sorties (`--out-dir`) :
- `pred_rfmid_finetune_val.npz` (format de run_baseline.py : comparaison
  appariée avec `results/pred_rfmid_crop-mask_val.npz` via
  scripts/compare_predictions.py) ;
- `pred_rfmid_finetune_test.npz` avec `--eval-test` (lu une seule fois) ;
- `pred_aptos_finetune.npz` avec `--aptos-root` : probabilité de
  rétinopathie diabétique sur les 3 662 images APTOS et leur format
  d'origine (analyse à appareil égal en local) ;
- `finetune_rfmid.json` : historique d'entraînement et métriques ;
- `finetune_rfmid.pt` : poids entraînés (blocs dégelés et tête).

Les images sont préparées une fois (`crop-mask`, 224 px) et gardées en
mémoire en uint8, pour ne pas refaire le recadrage à chaque epoch.

Exemple (Colab, GPU T4, ~30-45 min) :
    python scripts/finetune_retfound.py --data-root <RFMiD> \\
        --checkpoint third_party/retfound/checkpoints/RETFound_mae_natureCFP.pth \\
        --out-dir finetune --eval-test --aptos-root <APTOS>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.aptos_dataset import APTOSDataset  # noqa: E402
from src.data.fundus_preprocess import PREPROCESS_VERSION, common_mask  # noqa: E402
from src.data.rfmid_dataset import RFMiDDataset  # noqa: E402
from src.eval.metrics import evaluate_multilabel  # noqa: E402
from src.models.embedding_cache import file_fingerprint  # noqa: E402
from src.models.finetune import PartialFinetune, augment_batch, cosine_with_warmup  # noqa: E402
from src.models.retfound_backbone import load_pretrained_backbone  # noqa: E402

IMAGE_SIZE = 224


def preload(dataset, workers: int, limit: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Images en uint8 (N, 3, H, W) et étiquettes, préparées une seule fois."""
    if limit is not None:
        dataset = Subset(dataset, range(min(limit, len(dataset))))
    images, labels = [], []
    for x, y in DataLoader(dataset, batch_size=64, num_workers=workers):
        images.append((x * 255).round().to(torch.uint8))
        labels.append(y)
    return torch.cat(images), torch.cat(labels)


@torch.no_grad()
def predict(model, images: torch.Tensor, device: str, batch_size: int = 64) -> np.ndarray:
    model.eval()
    probs = []
    for i in range(0, len(images), batch_size):
        batch = images[i : i + batch_size].to(device).float() / 255
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
            probs.append(torch.sigmoid(model(batch).float()).cpu())
    return torch.cat(probs).numpy()


def save_predictions(path: Path, keys, groups, y, probs, class_names) -> None:
    np.savez(
        path, keys=np.array(keys), groups=np.array(groups), y_true=y, probs=probs, class_names=np.array(class_names)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, required=True, help="RFMiD")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--trainable-blocks", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr-backbone", type=float, default=2e-5)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eval-test", action="store_true", help="Lit le test RFMiD une fois, avec l'epoch retenue")
    parser.add_argument("--aptos-root", type=Path, default=None, help="Validation externe sur APTOS")
    parser.add_argument("--limit", type=int, default=None, help="Essai rapide : nombre d'images par jeu")
    parser.add_argument("--trust-checkpoint", action="store_true")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    t0 = time.time()

    sets = {s: RFMiDDataset(args.data_root, split=s, preprocess="crop-mask") for s in ("train", "val", "test")}
    class_names = sets["train"].class_names
    loaded = {s: preload(ds, args.workers, args.limit) for s, ds in sets.items() if s != "test" or args.eval_test}
    print(
        f"Images préparées en {time.time() - t0:.0f} s : " + ", ".join(f"{s} {len(x)}" for s, (x, _) in loaded.items())
    )

    vit = load_pretrained_backbone(
        args.checkpoint, img_size=IMAGE_SIZE, trust_checkpoint=True if args.trust_checkpoint else None
    )
    model = PartialFinetune(vit, len(class_names), args.trainable_blocks).to(args.device)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone_parameters(), "lr": args.lr_backbone},
            {"params": model.head.parameters(), "lr": args.lr_head},
        ],
        weight_decay=args.weight_decay,
    )
    base_lrs = [g["lr"] for g in optimizer.param_groups]
    use_amp = args.device.startswith("cuda")
    # torch.amp.GradScaler n'existe qu'à partir de torch 2.3 (requirements : >= 2.2)
    scaler = (
        torch.amp.GradScaler("cuda", enabled=use_amp)
        if hasattr(torch.amp, "GradScaler")
        else torch.cuda.amp.GradScaler(enabled=use_amp)
    )
    criterion = nn.BCEWithLogitsLoss()
    mask = torch.from_numpy(common_mask(IMAGE_SIZE))
    generator = torch.Generator().manual_seed(args.seed)

    trainable_names = [name for name, p in model.named_parameters() if p.requires_grad]
    X_train, y_train = loaded["train"]
    X_val, y_val = loaded["val"]
    steps_per_epoch = max(1, len(X_train) // args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    history, best, best_state, step, stale = [], -1.0, None, 0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(X_train), generator=generator)
        losses = []
        for i in range(steps_per_epoch):
            idx = order[i * args.batch_size : (i + 1) * args.batch_size]
            images = augment_batch(X_train[idx].float() / 255, mask, generator).to(args.device)
            labels = y_train[idx].to(args.device)
            factor = cosine_with_warmup(step, total_steps, steps_per_epoch)
            for group, lr in zip(optimizer.param_groups, base_lrs, strict=True):
                group["lr"] = lr * factor
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                loss = criterion(model(images).float(), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.item())
            step += 1
        val_auroc = evaluate_multilabel(y_val.numpy(), predict(model, X_val, args.device), class_names)["auroc_macro"]
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_auroc_macro": val_auroc})
        print(
            f"epoch {epoch:2d} | loss {np.mean(losses):.4f} | AUROC val {val_auroc:.4f} | {time.time() - t0:.0f} s",
            flush=True,
        )
        if val_auroc > best:
            best, stale = val_auroc, 0
            state = model.state_dict()
            best_state = {name: state[name].detach().cpu().clone() for name in trainable_names}
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Arrêt : {args.patience} epochs sans progrès")
                break

    model.load_state_dict(best_state, strict=False)
    best_epoch = max(history, key=lambda h: h["val_auroc_macro"])["epoch"]
    report = {
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "checkpoint_sha256": file_fingerprint(args.checkpoint),
        "preprocess_version": PREPROCESS_VERSION,
        "best_epoch": best_epoch,
        "history": history,
    }

    p_val = predict(model, X_val, args.device)
    keys_val = sets["val"].image_keys[: len(X_val)]
    save_predictions(
        args.out_dir / "pred_rfmid_finetune_val.npz", keys_val, keys_val, y_val.numpy(), p_val, class_names
    )
    report["val"] = evaluate_multilabel(y_val.numpy(), p_val, class_names)
    print(f"Epoch retenue {best_epoch} : AUROC macro validation {report['val']['auroc_macro']:.4f}")

    if args.eval_test:
        X_test, y_test = loaded["test"]
        p_test = predict(model, X_test, args.device)
        keys_test = sets["test"].image_keys[: len(X_test)]
        save_predictions(
            args.out_dir / "pred_rfmid_finetune_test.npz", keys_test, keys_test, y_test.numpy(), p_test, class_names
        )
        report["test"] = evaluate_multilabel(y_test.numpy(), p_test, class_names)
        print(f"Test (lu une fois) : AUROC macro {report['test']['auroc_macro']:.4f}")

    if args.aptos_root is not None:
        aptos = APTOSDataset(args.aptos_root, image_size=IMAGE_SIZE, preprocess="crop-mask")
        X_aptos, _ = preload(aptos, args.workers, args.limit)
        p_aptos = predict(model, X_aptos, args.device)[:, class_names.index("DR")]
        n = len(X_aptos)
        sizes = []
        for i in range(n):
            with Image.open(aptos.image_path(i)) as image:
                sizes.append(f"{image.width}x{image.height}")
        np.savez(
            args.out_dir / "pred_aptos_finetune.npz",
            keys=np.array(aptos.image_keys[:n]),
            grade=aptos.df["diagnosis"].to_numpy()[:n],
            prob_dr=p_aptos,
            formats=np.array(sizes),
        )
        any_dr = (aptos.df["diagnosis"].to_numpy()[:n] >= 1).astype(int)
        if 0 < any_dr.sum() < n:
            from sklearn.metrics import roc_auc_score

            report["aptos_any_dr_auroc"] = float(roc_auc_score(any_dr, p_aptos))
            print(f"APTOS, toute rétinopathie : AUROC {report['aptos_any_dr_auroc']:.4f}")

    torch.save(
        {"state_dict": best_state, "class_names": list(class_names), **report["config"]},
        args.out_dir / "finetune_rfmid.pt",
    )
    report["duration_s"] = time.time() - t0
    (args.out_dir / "finetune_rfmid.json").write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str))
    print(f"Terminé en {report['duration_s'] / 60:.0f} min -> {args.out_dir}")


if __name__ == "__main__":
    main()
