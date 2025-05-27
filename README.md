# Toric Code Simulator

This project provides a reproducible GPU-accelerated simulation of a 2‑D toric-code lattice using Python. The default configuration runs a small CPU-based sanity check so continuous integration can execute the unit tests without requiring a GPU.

## Requirements

Install the conda environment defined in `environment.yml`:

```bash
conda env create -f environment.yml
conda activate toric
```

If using an RTX 4090, ensure `CUDA_VISIBLE_DEVICES` is set to select the appropriate GPU. At least 24 GB of VRAM is recommended for large lattices.

## Running

Three common invocation examples are listed in the Run guide at the bottom of this document.

All results (CSV files, plots and pickled states) are written to `./results`.


## Run guide

- **CPU sanity run:**
  ```bash
  python main.py
  ```
- **32 × 32 GPU run:**
  ```bash
  python main.py --Lx 32 --Ly 32 --steps 10 --backend cuda
  ```
- **Black-hole Page-curve demo:**
  ```bash
  python main.py --hole --save_entropy
  ```
