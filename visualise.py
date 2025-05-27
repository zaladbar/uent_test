"""Plotting utilities."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_entropy(t: np.ndarray, entropy: np.ndarray, path: str) -> None:
    plt.figure()
    plt.plot(t, entropy)
    plt.xlabel("time")
    plt.ylabel("Entropy")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_page_curve(entropy: np.ndarray, path: str) -> None:
    plt.figure()
    plt.plot(entropy)
    plt.xlabel("step")
    plt.ylabel("Emitted entropy")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
