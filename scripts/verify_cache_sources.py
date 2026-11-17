#!/usr/bin/env python3
"""Contrôle (et pose) l'empreinte des images source dans les caches d'embeddings.

Constat 6 de l'audit du 15 novembre : un cache vérifiait le modèle, le prétraitement
et la taille, pas le contenu des images. Une image remplacée sous la même clé pouvait
réutiliser un embedding périmé. Les nouveaux embeddings portent désormais l'empreinte
de leur image (`EmbeddingCache.add(..., sources=...)`). Ce script :

- calcule l'empreinte des images présentes sur la machine pour chaque base couverte ;
- refuse (code de sortie 1) tout cache dont une image a changé depuis son empreinte ;
- avec `--adopt`, pose l'empreinte actuelle sur les embeddings qui n'en ont pas. Elle
  est marquée « supposée » image par image : rien ne prouve que l'embedding historique
  vient bien de ce fichier, mais tout changement ultérieur sera détecté.

Bases couvertes (images locales) : RFMiD, ODIR, ISIC 2019, Messidor-2, IDRiD (gradation),
ISIC 2020, MILK10k, PAD-UFES-20 (clé = nom du fichier, comme dans leurs scripts) et
SLICE-3D (octets de chaque image dans l'archive d'origine). Non couverts : images absentes
de la machine (APTOS, EyePACS, DDR) ; routeur et qualité, dont les entrées mêlent
plusieurs bases et des images dégradées fabriquées à la volée (pas de fichier source).

Exemple :
    python scripts/verify_cache_sources.py --data-root data/raw --adopt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.confirmation_external import idrid_table, messidor_table  # noqa: E402
from src.data.isic_dataset import ISIC2019Dataset  # noqa: E402
from src.data.odir_dataset import ODIRDataset  # noqa: E402
from src.data.rfmid_dataset import RFMiDDataset  # noqa: E402
from src.models.embedding_cache import CacheMismatchError, EmbeddingCache, file_fingerprint  # noqa: E402

SPLITS = ("train", "val", "test")


def dataset_paths(name: str, root: Path) -> dict[str, Path]:
    """Clé de cache -> fichier image, pour une base couverte."""
    if name == "rfmid":
        out = {}
        for split in SPLITS:
            ds = RFMiDDataset(root, split=split)
            out.update({k: ds.image_path(i) for i, k in enumerate(ds.image_keys)})
        return out
    if name == "odir":
        out = {}
        for split in SPLITS:
            ds = ODIRDataset(root, split=split)
            out.update({k: ds.image_path(i) for i, k in enumerate(ds.image_keys)})
        return out
    if name == "isic2019":
        out = {}
        for split in SPLITS:
            ds = ISIC2019Dataset(root, split=split)
            out.update({k: ds.image_path(i) for i, k in enumerate(ds.image_keys)})
        return out
    if name in BY_FILE_NAME:  # mêmes correspondances que les scripts d'analyse
        pattern, use_stem = BY_FILE_NAME[name]
        files = sorted(root.rglob(pattern))
        names = [p.stem if use_stem else p.name for p in files]
        if len(set(names)) != len(names):
            raise ValueError(f"{name} : noms d'images en double, correspondance ambiguë")
        return {f"{name}/{n}": p for n, p in zip(names, files, strict=True)}
    table = messidor_table(root) if name == "messidor2" else idrid_table(root)
    return dict(zip(table["key"], table["path"], strict=True))


# Bases dont la clé est « <base>/<nom du fichier> » (isic2020_external.py, skin_transfer_milk10k.py,
# pad_ufes20_skin.py) : motif, et nom sans extension (stem) ou complet.
BY_FILE_NAME = {"isic2020": ("*.jpg", True), "milk10k": ("*.jpg", True), "pad_ufes20": ("*.png", False)}
SLICE3D_ZIP = "ISIC_2024_Permissive_Training_Input.zip"


def slice3d_fingerprints(root: Path, keys: list[str]) -> dict[str, str]:
    """SLICE-3D : images lues dans l'archive (slice3d_feasibility.py) ; empreinte des octets du membre."""
    with zipfile.ZipFile(root / SLICE3D_ZIP) as archive:
        members = {Path(n).stem: n for n in archive.namelist() if n.endswith(".jpg")}
        wanted = [k for k in keys if k.startswith("slice3d/") and k.split("/", 1)[1] in members]
        return {k: hashlib.sha256(archive.read(members[k.split("/", 1)[1]])).hexdigest()[:16] for k in wanted}


DATASETS = {  # préfixe du fichier de cache -> (base, sous-dossier de data/raw)
    "rfmid_": ("rfmid", "rfmid"),
    "odir_": ("odir", "odir"),
    "isic2019_": ("isic2019", "isic2019"),
    "messidor2_": ("messidor2", "messidor2"),
    "idrid_": ("idrid", "idrid_grading"),
    "isic2020_": ("isic2020", "isic2020"),
    "milk10k_": ("milk10k", "milk10k"),
    "pad_ufes20_": ("pad_ufes20", "pad_ufes20"),
    "slice3d_": ("slice3d", "slice3d"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--caches", type=Path, nargs="+", default=[Path("embeddings"), Path("embeddings_derm")])
    parser.add_argument("--adopt", action="store_true", help="Poser l'empreinte actuelle (marquée supposée)")
    parser.add_argument("--out", type=Path, default=None, help="Bilan JSON (facultatif)")
    parser.add_argument("--only", nargs="+", default=None, help="Bases à contrôler (défaut : toutes)")
    args = parser.parse_args()

    fingerprints: dict[str, dict[str, str]] = {}
    report, failed = {}, False
    for path in sorted(p for d in args.caches if d.is_dir() for p in d.glob("*.npz")):
        match = next((v for prefix, v in DATASETS.items() if path.name.startswith(prefix)), None)
        if match is None:
            report[path.name] = {"status": "non couvert"}
            continue
        name, sub = match
        if args.only and name not in args.only:
            continue
        try:
            cache = EmbeddingCache(path)
            if name not in fingerprints:
                root = args.data_root / sub
                print(f"Empreintes des images {name} ({root})...", flush=True)
                if name == "slice3d":
                    fingerprints[name] = slice3d_fingerprints(root, list(cache.embeddings))
                else:
                    fingerprints[name] = {k: file_fingerprint(p) for k, p in dataset_paths(name, root).items()}
            counts = cache.check_sources(fingerprints[name], adopt=args.adopt)
            report[path.name] = {"status": "ok", **counts}
        except CacheMismatchError as error:
            report[path.name] = {"status": "IMAGES MODIFIÉES", "error": str(error)}
            failed = True
        print(f"  {path.name} : {report[path.name]}", flush=True)
    for name, entry in report.items():
        if entry["status"] == "non couvert":
            print(f"  {name} : non couvert")
    if args.out:
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
