"""Cache des embeddings RETFound, un fichier .npz par configuration.

Le backbone est figé : l'embedding d'une image ne dépend que de l'image, de
son prétraitement, de la taille et du checkpoint. Le recalculer à chaque run
coûtait ~10 min de GPU pour n'entraîner qu'une tête linéaire. Avec le cache
(sur Google Drive dans Colab), les expériences sur la tête tournent en
quelques secondes, sur CPU, et le backbone n'est même pas chargé.

Indexé par image (`dataset.image_keys`), pas par split : changer de split ou
d'étiquettes réutilise les embeddings déjà calculés.

Le nom du fichier ne suffit pas à garantir que les embeddings sont à jour :
un checkpoint remplacé sous le même nom, ou un prétraitement modifié, les
rendrait périmés sans erreur. Le cache enregistre donc une empreinte de sa
configuration (`meta` : empreinte du contenu du checkpoint, prétraitement et
sa version, taille d'image) et refuse de servir des embeddings calculés avec
une autre. Un ancien cache sans empreinte est adopté une fois, avec un
avertissement.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

ASSUMED_KEY = "assumed_not_verified"


class CacheMismatchError(RuntimeError):
    """Le cache a été calculé avec une autre configuration que celle demandée."""


def file_fingerprint(path: str | Path, chunk_size: int = 1 << 24) -> str:
    """Empreinte SHA-256 (16 premiers caractères) du contenu d'un fichier."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()[:16]


class EmbeddingCache:
    def __init__(self, path: str | Path, meta: dict | None = None):
        self.path = Path(path)
        self.meta = meta
        self.embeddings: dict[str, np.ndarray] = {}
        if not self.path.exists():
            return
        with np.load(self.path) as data:
            keys, X = data["keys"].tolist(), data["X"]
            stored = json.loads(str(data["meta"])) if "meta" in data.files else None
        if X.ndim != 2 or len(keys) != len(X):
            raise CacheMismatchError(f"Cache {self.path} mal formé : {len(keys)} clés pour un tableau {X.shape}")
        if len(set(keys)) != len(keys):
            raise CacheMismatchError(f"Cache {self.path} : {len(keys) - len(set(keys))} clé(s) en double")
        if not np.isfinite(X).all():
            raise CacheMismatchError(f"Cache {self.path} : valeurs non finies (NaN ou infini)")
        self.embeddings = dict(zip(keys, X, strict=True))
        if meta is None:
            self.meta = stored
            return
        if stored is None:
            print(
                f"⚠ Cache sans empreinte ({self.path.name}) : supposé calculé avec la configuration actuelle "
                f"{meta}. Le supprimer en cas de doute."
            )
            stored = {}
        # Une valeur demandée à None (ex. checkpoint absent de la machine) n'est pas vérifiée.
        conflicts = {
            k: (stored[k], v) for k, v in meta.items() if v is not None and stored.get(k) is not None and stored[k] != v
        }
        if conflicts:
            raise CacheMismatchError(
                f"Cache {self.path} calculé avec une autre configuration ({conflicts} : en cache, demandé). "
                "Supprimer ce fichier (ou changer de dossier de cache) pour recalculer les embeddings."
            )
        unchecked = [k for k, v in meta.items() if v is None and stored.get(k) is not None]
        if unchecked:
            print(f"⚠ Cache {self.path.name} : {unchecked} non vérifié(s) (valeur inconnue sur cette machine).")
        self.meta = {**stored, **{k: v for k, v in meta.items() if v is not None or k not in stored}}
        # Une valeur ajoutée ici est une déclaration, pas une preuve de l'origine
        # des embeddings : elle sert à détecter les changements et reste marquée.
        adopted = {k for k in meta if meta[k] is not None and stored.get(k) is None}
        if adopted or stored.get(ASSUMED_KEY):
            self.meta[ASSUMED_KEY] = sorted(adopted | set(stored.get(ASSUMED_KEY, [])))
        if self.meta != stored:
            self.save()  # enregistre l'empreinte complétée : les prochains changements seront détectés

    def missing(self, keys: list[str]) -> list[str]:
        return [k for k in keys if k not in self.embeddings]

    def add(self, keys: list[str], X: np.ndarray) -> None:
        self.embeddings.update(zip(keys, X.astype(np.float32), strict=True))

    def get(self, keys: list[str]) -> np.ndarray:
        return np.stack([self.embeddings[k] for k in keys])

    def save(self) -> None:
        """Écriture dans un fichier temporaire puis renommage : un run coupé
        (déconnexion Colab) ne laisse pas un cache à moitié écrit."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp.npz")
        keys = list(self.embeddings)
        extra = {} if self.meta is None else {"meta": np.array(json.dumps(self.meta, sort_keys=True))}
        np.savez(tmp, keys=np.array(keys), X=np.stack([self.embeddings[k] for k in keys]), **extra)
        os.replace(tmp, self.path)


def cache_file_name(dataset: str, preprocess: str, image_size: int, checkpoint: str | Path) -> str:
    return f"{dataset}_{preprocess}_{image_size}_{Path(checkpoint).stem}.npz"


# Nom de fichier du checkpoint de chaque modèle, et empreinte quand elle est connue
# sans le fichier (Green : release publique ; RETFound : poids pas toujours présents).
CHECKPOINT_STEMS = {"retfound": "RETFound_mae_natureCFP", "green": "retfoundgreen_statedict"}
KNOWN_CHECKPOINTS = {"green": "431de5dbc1bebbb3"}
NATIVE_SIZES = {"retfound": 224, "green": 392}


def load_embeddings(
    directory: str | Path,
    dataset: str,
    keys: list[str],
    backbone: str = "retfound",
    preprocess: str = "crop-mask",
    image_size: int | None = None,
) -> np.ndarray:
    """Chargeur commun des analyses : refuse un cache calculé avec une autre
    configuration (prétraitement, sa version, taille, checkpoint quand il est
    connu) ou auquel manquent des images."""
    from src.data.fundus_preprocess import PREPROCESS_VERSION

    if backbone not in CHECKPOINT_STEMS:
        raise ValueError(f"modèle {backbone!r} inconnu, attendu parmi {sorted(CHECKPOINT_STEMS)}")
    size = image_size or NATIVE_SIZES[backbone]
    meta = {
        "checkpoint_sha256": KNOWN_CHECKPOINTS.get(backbone),
        "preprocess": preprocess,
        "preprocess_version": PREPROCESS_VERSION,
        "image_size": size,
    }
    path = Path(directory) / f"{dataset}_{preprocess}_{size}_{CHECKPOINT_STEMS[backbone]}.npz"
    cache = EmbeddingCache(path, meta=meta)
    missing = cache.missing(keys)
    if missing:
        raise CacheMismatchError(f"{path.name} : {len(missing)} image(s) absente(s), dont {missing[:3]}")
    return cache.get(keys).astype(np.float32)
