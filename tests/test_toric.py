import numpy as np
from hamiltonian import build_stabilizers, stabilizer_commute, codespace_dimension, Stabilizer


def test_ground_state_degeneracy():
    assert codespace_dimension(2, 2) == 4


def test_commutation():
    stars, plaquettes = build_stabilizers(2, 2)
    for a in stars + plaquettes:
        for b in stars + plaquettes:
            assert stabilizer_commute(a, b)


def test_single_qubit_syndrome():
    lx = ly = 2
    stars, plaquettes = build_stabilizers(lx, ly)
    n = 2 * lx * ly
    # apply Z error on qubit 0
    err_z = Stabilizer(np.zeros(n, dtype=np.uint8), np.eye(1, n, dtype=np.uint8)[0])
    anti = [s for s in stars if not stabilizer_commute(s, err_z)]
    assert len(anti) == 2
    # apply X error on qubit 0
    err_x = Stabilizer(np.eye(1, n, dtype=np.uint8)[0], np.zeros(n, dtype=np.uint8))
    anti = [p for p in plaquettes if not stabilizer_commute(p, err_x)]
    assert len(anti) == 2
