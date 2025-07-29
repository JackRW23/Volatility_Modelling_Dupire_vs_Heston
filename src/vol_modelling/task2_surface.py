# ---------------------------------------------------------------------------
#  src/vol_modelling/task2_surface.py               (rewritten 2025-04-23)
# ---------------------------------------------------------------------------
"""
Task 2 – build a Heston European-call price surface and save both a 2-D
heat-map and a 3-D surface render.

Plot upgrade
------------
* Uses the “turbo” colormap (matches Tasks 3 & 4).
* Colour limits are chosen automatically from the data (1st / 99th
  percentiles) rather than hard-coded.
* 3-D surface has a thin black wire-frame for better depth perception.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – registers 3-D projection

# (Param Centralization) import parameters for global config values
import vol_modelling.parameters as parameters
from vol_modelling.common import build_price_surface, ensure_dir

OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "plots"
DATA_DIR   = Path(__file__).resolve().parent.parent.parent / "data"

# ---------------------------------------------------------------------------
#  Plot helpers
# ---------------------------------------------------------------------------
def _get_colour_limits(Z: np.ndarray) -> tuple[float, float]:
    """Robust vmin / vmax = 1st & 99th percentiles of finite values in Z."""
    finite_vals = Z[np.isfinite(Z)]
    vmin = float(np.nanpercentile(finite_vals, 1.0))
    vmax = float(np.nanpercentile(finite_vals, 99.0))
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-3
    return vmin, vmax


def _plot_heatmap(X, Y, Z, title: str, cbar_label: str) -> None:
    vmin, vmax = _get_colour_limits(Z)

    plt.figure(figsize=(8, 6))
    plt.pcolormesh(
        X,
        Y,
        Z,
        shading="auto",
        cmap="turbo",
        vmin=vmin,
        vmax=vmax,
    )
    plt.colorbar(label=cbar_label)
    plt.xlabel("Strike")
    plt.ylabel("Maturity (years)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "task2_price_surface.png", dpi=150)
    plt.close()


def _plot_surface3d(X, Y, Z, title: str, cbar_label: str) -> None:
    vmin, vmax = _get_colour_limits(Z)

    fig = plt.figure(figsize=(8, 6))
    ax  = fig.add_subplot(projection="3d")
    surf = ax.plot_surface(
        X,
        Y,
        Z,
        cmap="turbo",
        rstride=1,
        cstride=1,
        linewidth=0.25,        # thin black mesh
        edgecolor="k",
        antialiased=True,
        vmin=vmin,
        vmax=vmax,
    )
    fig.colorbar(surf, shrink=0.7, aspect=12, label=cbar_label)
    ax.set_xlabel("Strike")
    ax.set_ylabel("Maturity (years)")
    ax.set_zlabel(cbar_label)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "task2_price_surface_3d.png", dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
#  Driver – run Task 2
# ---------------------------------------------------------------------------
def run_task() -> None:
    """Compute price grid, save CSV, and output both 2-D and 3-D figures."""
    params     = parameters.current_params
    maturities = np.linspace(
        parameters.MATURITY_MIN,
        parameters.MATURITY_MAX,
        parameters.GRID_SIZE,
    )
    strikes = np.linspace(
        parameters.STRIKE_MIN,
        parameters.STRIKE_MAX,
        parameters.GRID_SIZE,
    )

    price_df = build_price_surface(params, maturities, strikes)

    ensure_dir(DATA_DIR)
    ensure_dir(OUTPUT_DIR)
    price_df.to_csv(DATA_DIR / "heston_call_surface.csv")

    X, Y = np.meshgrid(price_df.columns.astype(float), price_df.index.astype(float))
    Z    = price_df.values

    _plot_heatmap(
        X,
        Y,
        Z,
        "Heston call-price surface (Task 2)",
        "Call price",
    )
    _plot_surface3d(
        X,
        Y,
        Z,
        "Heston call-price surface (Task 2, 3-D)",
        "Call price",
    )

    print(
        f"[Task 2]  Surface built – saved 2-D and 3-D plots "
        f"({price_df.shape[0]}×{price_df.shape[1]})"
    )


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_task()
