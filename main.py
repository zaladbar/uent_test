"""Command line interface for toric code simulation."""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from backend_cpu import CPUBackend
from backend_cuda import CuQuantumBackend
from hamiltonian import build_stabilizers
from visualise import plot_entropy


RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def run_simulation(lx: int, ly: int, steps: int, backend: str, hole: bool = False) -> None:
    stars, plaquettes = build_stabilizers(lx, ly)
    if hole and plaquettes:
        plaquettes.pop(len(plaquettes) // 2)

    BackendCls = CPUBackend if backend == "cpu" else CuQuantumBackend
    sim = BackendCls(num_qubits=2 * lx * ly)

    ent = []
    for t in range(steps):
        sim.evolve(stars + plaquettes, dt=0.1)
        ent.append(sim.entropy(list(range(sim.num_qubits // 2))))

    plot_entropy(
        np.arange(steps),
        np.array(ent),
        str(RESULTS_DIR / "entropy.png"),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--Lx", type=int, default=4, help="lattice size x")
    p.add_argument("--Ly", type=int, default=4, help="lattice size y")
    p.add_argument("--steps", type=int, default=1, help="number of trotter steps")
    p.add_argument("--backend", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--hole", action="store_true", help="punch a missing plaquette")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_simulation(args.Lx, args.Ly, args.steps, args.backend, args.hole)


if __name__ == "__main__":
    main()
