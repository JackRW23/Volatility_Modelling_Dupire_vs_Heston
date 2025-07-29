# ---------------------------------------------------------------------------
#  src/vol_modelling/task3_iv.py
# ---------------------------------------------------------------------------

"""
Task 3 – compute the implied-volatility surface from Task 2’s price grid and
produce both a 2-D heat-map and a 3-D surface plot.

Colour scheme:
    • Uses Matplotlib’s “turbo” colormap (a modern, perceptually uniform
      rainbow that mimics the colours in the reference image).
    • Adds a thin black wire-frame on the 3-D surface so the mesh is visible,
      again matching the look of the example plot.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# (Param Centralization) import global parameters
import vol_modelling.parameters as parameters
from vol_modelling.common import build_iv_surface, ensure_dir

# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
PLOT_DIR = Path(__file__).resolve().parent.parent.parent / "plots"

# ---------------------------------------------------------------------------


def _plot_heatmap(X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> None:
    """2-D heat-map of the implied-volatility surface."""
    plt.figure(figsize=(8, 6))
    # “turbo” ≈ rainbow/jet but perceptually uniform (Matplotlib ≥ 3.3).
    plt.pcolormesh(X, Y, Z, shading="auto", cmap="turbo")
    plt.colorbar(label="Implied vol")
    plt.xlabel("Strike")
    plt.ylabel("Maturity (years)")
    plt.title("Implied-volatility surface (Task 3)")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "task3_iv_surface.png", dpi=150)
    plt.close()


def _plot_surface3d(X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> None:
    """3-D surface render of the implied-volatility surface."""
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(projection="3d")

    # Draw the surface with the same vivid colour sweep and a fine wire-frame.
    surf = ax.plot_surface(
        X,
        Y,
        Z,
        cmap="turbo",
        rstride=1,
        cstride=1,
        linewidth=0.25,      # thin grid lines
        edgecolor="k",
        antialiased=True,
    )

    fig.colorbar(surf, shrink=0.7, aspect=12, label="Implied vol")
    ax.set_xlabel("Strike")
    ax.set_ylabel("Maturity (years)")
    ax.set_zlabel("Implied vol")
    ax.set_title("Implied-volatility surface (Task 3, 3-D)")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "task3_iv_surface_3d.png", dpi=150)
    plt.close()


# ---------------------------------------------------------------------------


def run_task() -> None:
    """Derive the IV surface, save it to CSV, and output both plots."""
    ensure_dir(DATA_DIR)
    ensure_dir(PLOT_DIR)

    price_csv = DATA_DIR / "heston_call_surface.csv"
    if not price_csv.exists():
        raise FileNotFoundError("Price surface CSV not found – run Task 2 first.")

    price_df = pd.read_csv(price_csv, index_col=0)
    iv_df = build_iv_surface(price_df, parameters.current_params)  # (Param Centralization)
    iv_df.to_csv(DATA_DIR / "heston_iv_surface.csv")

    # Build mesh-grids for plotting
    X, Y = np.meshgrid(iv_df.columns.astype(float), iv_df.index.astype(float))
    Z = iv_df.values

    _plot_heatmap(X, Y, Z)
    _plot_surface3d(X, Y, Z)

    nan_ct = np.isnan(Z).sum()
    print(f"[Task 3]  IV surface saved – NaNs: {nan_ct}")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_task()
