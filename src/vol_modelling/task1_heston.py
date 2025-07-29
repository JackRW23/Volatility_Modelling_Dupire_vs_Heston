# ---------------------------------------------------------------------------
#  src/vol_modelling/task1_heston.py
# ---------------------------------------------------------------------------

"""Task 1  -  Simulate Heston spot/variance paths and basic diagnostics."""

from __future__ import annotations

import matplotlib.pyplot as plt
from pathlib import Path

# (Param Centralization) import global parameters instead of local constants
import vol_modelling.parameters as parameters

from vol_modelling.common import (
    build_heston_process,
    simulate_heston_paths,
    ensure_dir,
)

OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "plots"

def run_task(n_paths: int = 1_000,
             maturity: float = 1.0,
             n_steps: int = 252,
             seed: int = 42) -> None:
    """Run Task 1 (Heston path simulation) and save a diagnostic plot."""
    params = parameters.current_params  # Use global Heston parameters
    t, S, V = simulate_heston_paths(build_heston_process(params), maturity, n_steps, n_paths, seed)

    # --- plot first 20 paths ------------------------------------------------
    ensure_dir(OUTPUT_DIR)
    plt.figure(figsize=(8, 4))
    plt.plot(t, S[:20].T)
    plt.xlabel("Time (years)")
    plt.ylabel("$S_t$")
    plt.title("Heston spot paths — first 20 of {:d}".format(n_paths))
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "task1_heston_paths.png", dpi=150)
    plt.close()

    # quick console diagnostics
    mean_S, std_S = S[:, -1].mean(), S[:, -1].std()
    print(f"[Task 1]  E[S(T)] = {mean_S:.4f},  SD[S(T)] = {std_S:.4f}")
    return mean_S, std_S

if __name__ == "__main__":
    run_task()