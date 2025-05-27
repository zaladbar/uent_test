"""Toric-code Hamiltonian utilities."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class Stabilizer:
    """Representation of a Pauli stabilizer in binary tableau form."""

    x: np.ndarray  # shape (n,)
    z: np.ndarray  # shape (n,)

    @property
    def n(self) -> int:
        return self.x.size

    def to_tableau(self) -> np.ndarray:
        return np.concatenate([self.x, self.z])


def _edge_index(lx: int, ly: int, x: int, y: int, vertical: bool) -> int:
    if vertical:
        return lx * ly + y * lx + x
    return y * lx + x


def build_stabilizers(lx: int, ly: int) -> Tuple[List[Stabilizer], List[Stabilizer]]:
    """Construct star and plaquette stabilizers for a periodic ``lx`` × ``ly`` lattice."""
    n_edges = 2 * lx * ly
    stars: List[Stabilizer] = []
    plaquettes: List[Stabilizer] = []

    for y in range(ly):
        for x in range(lx):
            xv = np.zeros(n_edges, dtype=np.uint8)
            zv = np.zeros(n_edges, dtype=np.uint8)
            # star operator uses Pauli X on four incident edges
            for dx, dy, vert in [(0, y, False), (x, y, True), ((x - 1) % lx, y, False), (x, (y - 1) % ly, True)]:
                idx = _edge_index(lx, ly, dx % lx, dy % ly, vert)
                xv[idx] = 1
            stars.append(Stabilizer(xv, zv))

    for y in range(ly):
        for x in range(lx):
            xv = np.zeros(n_edges, dtype=np.uint8)
            zv = np.zeros(n_edges, dtype=np.uint8)
            # plaquette uses Pauli Z on four surrounding edges
            for dx, dy, vert in [
                (x, y, False),
                (x, (y + 1) % ly, True),
                ((x + 1) % lx, y, False),
                (x, y, True),
            ]:
                idx = _edge_index(lx, ly, dx % lx, dy % ly, vert)
                zv[idx] = 1
            plaquettes.append(Stabilizer(xv, zv))
    return stars, plaquettes


def stabilizer_commute(a: Stabilizer, b: Stabilizer) -> bool:
    """Return True if two stabilizers commute under the symplectic product."""
    return bool((a.x @ b.z + a.z @ b.x) % 2 == 0)


def codespace_dimension(lx: int, ly: int) -> int:
    """Return the expected ground-state degeneracy for a toric code."""
    n_edges = 2 * lx * ly
    stars, plaquettes = build_stabilizers(lx, ly)
    rank = len(stars) + len(plaquettes) - 2
    return 2 ** (n_edges - rank)
