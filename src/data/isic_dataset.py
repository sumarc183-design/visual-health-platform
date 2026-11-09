"""ISIC 2019 (dermoscopie, CC BY-NC) : seconde indication.

25 331 images d'entraînement, une classe par image parmi 8 (mélanome, nævus,
carcinome basocellulaire, kératose actinique, kératose bénigne, dermatofibrome,
lésion vasculaire, carcinome épidermoïde), et un test officiel de 8 238 images dont
2 047 hors de ces classes (« UNK »), écartées ici.

Beaucoup de lésions sont photographiées plusieurs fois (16 459 images sur 25 331) :
groupes = identifiant de lésion, ou l'image quand il manque. La validation est tirée
dans l'entraînement en groupes (StratifiedGroupKFold, 1 pli sur 7, graine fixe). Le
test officiel ne donne pas d'identifiant de lésion : bootstrap par image.

Structure attendue sous `root` : ISIC_2019_Training_Input/, ISIC_2019_Test_Input/
(images .jpg) et les quatre CSV officiels (GroundTruth et Metadata, Training et Test).

Images : protocole d'évaluation de PanDerm (petit côté à 256 px pour une sortie de
224, puis recadrage central), le même pour tous les modèles comparés. Les images
ISIC ne sont pas carrées (600 × 450, 1 024 × 768...) : un redimensionnement simple
les déformerait.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset

CLASS_NAMES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
VALIDATION_FOLDS = 7
RESIZE_RATIO = 256 / 224


def load_skin_image(path: str | Path, image_size: int) -> torch.Tensor:
    """Image (3, size, size) dans [0, 1] : petit côté à round(size × 256/224)
    (bilinéaire), puis recadrage central, comme `Resize(256), CenterCrop(224)`."""
    image = Image.open(path).convert("RGB")
    short = round(image_size * RESIZE_RATIO)
    scale = short / min(image.size)
    w, h = max(short, round(image.width * scale)), max(short, round(image.height * scale))
    image = image.resize((w, h), Image.BILINEAR)
    left, top = (w - image_size) // 2, (h - image_size) // 2
    image = image.crop((left, top, left + image_size, top + image_size))
    return torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).permute(2, 0, 1)


class ISIC2019Dataset(Dataset):
    def __init__(self, root: str | Path, split: str = "train", image_size: int = 224, seed: int = 0):
        if split not in ("train", "val", "test"):
            raise ValueError(f"split doit être train/val/test, reçu {split!r}")
        self.root = Path(root)
        self.image_size = image_size
        self.class_names = CLASS_NAMES
        part = "Test" if split == "test" else "Training"
        labels = pd.read_csv(self.root / f"ISIC_2019_{part}_GroundTruth.csv")
        meta = pd.read_csv(self.root / f"ISIC_2019_{part}_Metadata.csv")
        df = labels.merge(meta, on="image", how="left", validate="one_to_one")
        if "UNK" in df:
            df = df[df["UNK"] != 1].reset_index(drop=True)
        if (df[CLASS_NAMES].sum(axis=1) != 1).any():
            raise ValueError("chaque image doit avoir exactement une des 8 classes")
        lesion = df["lesion_id"] if "lesion_id" in df else pd.Series([np.nan] * len(df))
        df["group"] = lesion.where(lesion.notna(), df["image"]).astype(str)
        if split != "test":
            y = df[CLASS_NAMES].to_numpy().argmax(axis=1)
            splitter = StratifiedGroupKFold(VALIDATION_FOLDS, shuffle=True, random_state=seed)
            fit, held_out = next(splitter.split(df, y, groups=df["group"]))
            df = df.iloc[held_out if split == "val" else fit].reset_index(drop=True)
        self.df = df
        self.images_dir = self.root / f"ISIC_2019_{part}_Input"

    @property
    def image_keys(self) -> list[str]:
        return [f"isic2019/{i}" for i in self.df["image"]]

    @property
    def groups(self) -> list[str]:
        return list(self.df["group"])

    def image_path(self, idx: int) -> Path:
        return self.images_dir / f"{self.df.iloc[idx]['image']}.jpg"

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        image = load_skin_image(self.image_path(idx), self.image_size)
        labels = torch.tensor(self.df.iloc[idx][CLASS_NAMES].to_numpy(dtype=np.float32))
        return image, labels
