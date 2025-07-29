# ---------------------------------------------------------------------------
#  src/vol_modelling/task5_reprice.py
# ---------------------------------------------------------------------------

"""Task 5 - re-price an off-grid European call at future valuation times
using the Dupire local-vol surface (calibrated at t=0) and compare with the
"true" analytic Heston price."""  

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# (Param Centralization) import global parameters
import vol_modelling.parameters as parameters

from vol_modelling.common import (
    build_heston_process,
    simulate_heston_paths,
    ensure_dir,
)
from vol_modelling.task4_dupire import (
    DupireSurface,
    price_call_local_vol,
    heston_price_given_spot,
)

# Constants & I/O
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
PLOT_DIR = ROOT_DIR / "plots"

K_TEST = parameters.TEST_STRIKE    # (Param Centralization)
T_TEST = parameters.TEST_MATURITY  # (Param Centralization)
SNAP_TIMES = np.array(parameters.SNAPSHOT_TIMES)  # (Param Centralization)

def run_task() -> None:
    ensure_dir(PLOT_DIR)

    # Rebuild Dupire surface from Task 3 IV CSV
    iv_csv = DATA_DIR / "heston_iv_surface.csv"
    if not iv_csv.exists():
        raise FileNotFoundError("IV surface CSV not found - run Task 3 first.")
    iv_df = pd.read_csv(iv_csv, index_col=0)
    maturities = iv_df.index.astype(float).values
    strikes    = iv_df.columns.astype(float).values
    lv = DupireSurface(strikes, maturities, iv_df.values.astype(float))

    # Simulate Heston paths out to maturity
    params = parameters.current_params  # (Param Centralization)
    n_mc = 1_000
    t_grid, S_paths, _ = simulate_heston_paths(
        build_heston_process(params), T_TEST, 252, n_mc, seed=777
    )
    # Map snapshot times to nearest index in t_grid
    idx = [int(np.searchsorted(t_grid, tau)) for tau in SNAP_TIMES]

    dup_prices: List[float] = []
    dup_err:    List[float] = []
    true_prices: List[float] = []

    for col, tau in enumerate(SNAP_TIMES):
        S_now = S_paths[:, idx[col]].mean()
        dup_px, dup_se = price_call_local_vol(lv, S_now, K_TEST, params.r, T_TEST - tau, 252, parameters.MC_PATHS_TASK5)  # (Param Centralization)
        dup_prices.append(dup_px); dup_err.append(dup_se)
        heston_px = heston_price_given_spot(params, S_now, K_TEST, T_TEST - tau)
        true_prices.append(heston_px)

    # Plot results
    plt.figure(figsize=(8, 4))
    plt.plot(SNAP_TIMES, true_prices, "o-", label="Heston true")
    plt.errorbar(SNAP_TIMES, dup_prices, yerr=dup_err, fmt="s-", capsize=4, label="Dupire repriced")
    # flip the x‑axis so “far from maturity” is on the left
    plt.gca().invert_xaxis()          #  ← add this line
    plt.xlabel("Time before maturity (years)")
    plt.ylabel("Option price")
    plt.title("Task 5 - Dupire vs. Heston repricing")
    plt.legend(); plt.tight_layout()
    plt.savefig(PLOT_DIR / "task5_repricing.png", dpi=150); plt.close()

    # Console summary of results
    diffs = np.array(dup_prices) - np.array(true_prices)
    print(f"[Task 5] Price diffs (Dupire - Heston): " + ", ".join(f"{d:.5f}" for d in diffs))

if __name__ == "__main__":
    run_task()