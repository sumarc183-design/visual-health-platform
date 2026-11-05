"""Doublons d'images repérés par leurs embeddings, et groupes indépendants.

Une même photo (ou presque) présente plusieurs fois dans un jeu fausse les
évaluations si ses copies tombent de part et d'autre d'un découpage
(ajustement / évaluation, plis de validation croisée) ou sont tirées
séparément par le bootstrap. Le second audit (5 novembre 2026) a trouvé dans
APTOS 122 groupes de vecteurs exactement identiques (249 images).

- `exact_duplicate_groups` : vecteurs identiques au bit près (empreinte
  SHA-256 du vecteur float32).
- `near_duplicate_pairs` : paires de similarité cosinus >= seuil (photo
  recompressée, légèrement recadrée...), à inspecter.
- `merge_groups` : fusionne plusieurs partitions (patient, doublons) en
  composantes connexes : deux images liées par l'une ou l'autre restent dans
  le même groupe.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict

import numpy as np


def exact_duplicate_groups(X: np.ndarray) -> np.ndarray:
    """Identifiant de groupe par ligne : lignes identiques -> même groupe."""
    first = {}
    groups = np.empty(len(X), dtype=np.int64)
    for i, row in enumerate(np.ascontiguousarray(X, dtype=np.float32)):
        digest = hashlib.sha256(row.tobytes()).hexdigest()
        groups[i] = first.setdefault(digest, i)
    return groups


def near_duplicate_pairs(X: np.ndarray, threshold: float = 0.995, block: int = 1024) -> list[tuple[int, int]]:
    """Paires (i, j), i < j, de similarité cosinus >= threshold."""
    Z = X / np.linalg.norm(X, axis=1, keepdims=True)
    pairs = []
    for start in range(0, len(Z), block):
        sim = Z[start : start + block] @ Z.T
        rows, cols = np.nonzero(sim >= threshold)
        pairs += [(start + r, c) for r, c in zip(rows, cols, strict=True) if start + r < c]
    return pairs


def merge_groups(n: int, *partitions, pairs: list[tuple[int, int]] = ()) -> list[str]:
    """Composantes connexes : lignes de même étiquette dans une des partitions,
    ou reliées par une paire, -> même groupe. Renvoie une étiquette par ligne."""
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        parent[find(i)] = find(j)

    for labels in partitions:
        members = defaultdict(list)
        for i, label in enumerate(labels):
            members[label].append(i)
        for rows in members.values():
            for r in rows[1:]:
                union(rows[0], r)
    for i, j in pairs:
        union(i, j)
    return [f"g{find(i)}" for i in range(n)]


def duplicate_report(groups: list[str], grades: np.ndarray) -> dict:
    """Combien de groupes de plus d'une image, et combien aux grades contradictoires."""
    members = defaultdict(list)
    for i, g in enumerate(groups):
        members[g].append(i)
    multi = [rows for rows in members.values() if len(rows) > 1]
    return {
        "groups_with_several_images": len(multi),
        "images_in_such_groups": int(sum(len(r) for r in multi)),
        "groups_with_different_grades": int(sum(len(set(grades[r])) > 1 for r in multi)),
        "groups_with_contradictory_any_dr": int(sum(len(set(grades[r] >= 1)) > 1 for r in multi)),
    }


NEAR_DUPLICATE_COSINE = 0.999  # au-dessous, des images différentes se relient (embeddings RETFound très semblables)


def independent_groups(X: np.ndarray, base_groups: list[str], near_threshold: float = NEAR_DUPLICATE_COSINE):
    """Groupes indépendants : patient (ou image) fusionné avec les doublons
    exacts et quasi exacts. Renvoie (étiquettes de groupe, nombre de paires
    quasi exactes)."""
    pairs = near_duplicate_pairs(X, near_threshold)
    return merge_groups(len(X), base_groups, exact_duplicate_groups(X), pairs=pairs), len(pairs)
