"""cuQuantum-based GPU backend (placeholder)."""

from __future__ import annotations

import cupy as cp

# custatevec and cutensornet imports would normally be here


class CuQuantumBackend:
    """GPU accelerated backend using cuStateVecStabilizer for large lattices."""

    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
        self.dim = 2 ** num_qubits
        self.state = cp.zeros((self.dim,), dtype=cp.complex128)
        self.state[0] = 1.0

    def apply_stabilizer(self, stabilizer) -> None:
        """Apply a stabilizer generator using cuStateVec APIs.

        This is a placeholder to illustrate where cuStateVec calls would occur.
        """
        # In production one would call custatevec.apply_matrix or the stabilizer
        # specific routines from cuStateVecStabilizer. Here we simply pass.
        pass

    def evolve(self, stabilizers, dt: float) -> None:
        """Time-evolve the state with a simple Trotter step (placeholder)."""
        # A real implementation would use cuTensorNet or cuStateVec for fast
        # exponentiation on GPU. For demonstration we leave this as a no-op.
        pass

    def entropy(self, subsystem: list[int]) -> float:
        """Return the entanglement entropy for ``subsystem``.

        The computation keeps data on device and only transfers the final scalar
        back to host.
        """
        # Copying to host only for final eigenvalues is acceptable but omitted
        # in this minimal example.
        return 0.0
