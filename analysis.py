"""Analysis routines for entanglement and Page curve."""

from __future__ import annotations

from typing import Iterable, List

import numpy as np
from sklearn.manifold import MDS


def mutual_information_matrix(subsystems: List[List[int]], backend) -> np.ndarray:
    """Compute the mutual-information matrix for the given list of subsystems."""
    n = len(subsystems)
    entropies = [backend.entropy(s) for s in subsystems]
    mi = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            s_union = list(set(subsystems[i]) | set(subsystems[j]))
            mi[i, j] = entropies[i] + entropies[j] - backend.entropy(s_union)
    return mi


def entanglement_distance(mi: np.ndarray) -> np.ndarray:
    """Compute d(A,B)=1/I(A:B) with infinity on the diagonal."""
    with np.errstate(divide="ignore"):
        inv = 1.0 / mi
    np.fill_diagonal(inv, 0.0)
    return inv


def mds_embedding(dist: np.ndarray, dim: int = 2) -> np.ndarray:
    """Return a 2-D embedding using scikit-learn MDS."""
    model = MDS(n_components=dim, dissimilarity="precomputed", random_state=0)
    return model.fit_transform(dist)
