"""CPU reference backend for small lattices."""

from __future__ import annotations

import numpy as np
from numpy import ndarray
from scipy.linalg import expm

from hamiltonian import Stabilizer


class CPUBackend:
    """State-vector simulator using NumPy."""

    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
        self.dim = 2 ** num_qubits
        # |0...0>
        self.state = np.zeros((self.dim,), dtype=complex)
        self.state[0] = 1.0

    def _pauli_matrix(self, st: Stabilizer) -> ndarray:
        mats = []
        for x, z in zip(st.x, st.z):
            if x and z:
                mats.append(np.array([[0, -1j], [1j, 0]], dtype=complex))  # Y
            elif x:
                mats.append(np.array([[0, 1], [1, 0]], dtype=complex))  # X
            elif z:
                mats.append(np.array([[1, 0], [0, -1]], dtype=complex))  # Z
            else:
                mats.append(np.eye(2, dtype=complex))
        out = mats[0]
        for m in mats[1:]:
            out = np.kron(out, m)
        return out

    def apply_stabilizer(self, st: Stabilizer) -> None:
        op = self._pauli_matrix(st)
        self.state = op @ self.state

    def evolve(self, stabilizers: list[Stabilizer], dt: float) -> None:
        h = sum(self._pauli_matrix(st) for st in stabilizers)
        u = expm(-1j * dt * h)
        self.state = u @ self.state

    def entropy(self, subsystem: list[int]) -> float:
        dim_a = 2 ** len(subsystem)
        dim_b = 2 ** (self.num_qubits - len(subsystem))
        state = self.state.reshape([2] * self.num_qubits)
        axes = tuple(i for i in range(self.num_qubits) if i not in subsystem)
        psi = np.reshape(np.moveaxis(state, subsystem + axes, range(self.num_qubits)), (dim_a, dim_b))
        rho = psi @ psi.conj().T
        vals = np.linalg.eigvalsh(rho)
        vals = vals[vals > 1e-12]
        return float(-np.sum(vals * np.log2(vals)))
