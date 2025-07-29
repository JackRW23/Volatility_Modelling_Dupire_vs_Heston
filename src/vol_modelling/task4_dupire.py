# ---------------------------------------------------------------------------
#  src/vol_modelling/task4_dupire.py              (rewritten 2025-04-23 b)
# ---------------------------------------------------------------------------
"""
Task 4 – build the Dupire local-volatility surface from the implied-vol grid
(Task 3) and test it by Monte-Carlo repricing of an off-grid European call.

Colour-bar fix
--------------
* The vmin / vmax used for every plot are now **computed from the data**:
  – we take the 1 %- and 99 %-percentiles of the local-vol matrix to avoid
    rare outliers flattening the colour range.
  – both the 2-D heat-map and the 3-D surface share the same limits so the
    colours correspond one-for-one.
* “turbo” colormap + black wire-frame (unchanged from the previous commit).

Other key improvements remain unchanged:
* Box-clamped bilinear interpolation,
* Log-Euler path simulation,
* Runtime warnings on extreme vol,
*   AnalyticHestonEngine integration order capped at 192.
"""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import Tuple
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import QuantLib as ql

from vol_modelling.common import (
    HestonParams,
    ensure_dir,
    build_heston_model,
    date_from_years,
)
from vol_modelling import parameters

# ---------------------------------------------------------------------------
#  Constants & I/O
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
PLOT_DIR = ROOT_DIR / "plots"

# ---------------------------------------------------------------------------
#  Finite-difference helper
# ---------------------------------------------------------------------------
def _central_diff(arr: np.ndarray, axis: int, h: float) -> np.ndarray:
    """Second-order central difference with one-sided boundaries."""
    fwd   = np.roll(arr, -1, axis=axis)
    bwd   = np.roll(arr,  1, axis=axis)
    deriv = (fwd - bwd) / (2.0 * h)

    if axis == 0:                       # time derivative
        deriv[0]   = (arr[1] - arr[0]) / h
        deriv[-1]  = (arr[-1] - arr[-2]) / h
    else:                               # strike derivative
        deriv[:, 0]  = (arr[:, 1] - arr[:, 0]) / h
        deriv[:, -1] = (arr[:, -1] - arr[:, -2]) / h
    return deriv

# ---------------------------------------------------------------------------
#  Dupire local-volatility surface
# ---------------------------------------------------------------------------
@dataclass
class DupireSurface:
    strikes:     np.ndarray        # shape (M,)
    maturities:  np.ndarray        # shape (N,)
    sigma_iv:    np.ndarray        # shape (N, M) – rows = T, cols = K

    # computed in __post_init__
    local_vol:   np.ndarray = None
    _interp:     ql.BilinearInterpolation = None

    def __post_init__(self):
        self._build_local_vol()

    # ---------------------------------------------------------------------
    #  internal utilities
    # ---------------------------------------------------------------------
    def _clamp(self, S: float, t: float) -> tuple[float, float]:
        """Keep (S,t) within the calibrated strike / maturity box."""
        S_lo, S_hi = self.strikes[0],    self.strikes[-1]
        t_lo, t_hi = self.maturities[0], self.maturities[-1]
        return (
            min(max(S, S_lo), S_hi),
            min(max(t, t_lo), t_hi),
        )

    # ---------------------------------------------------------------------
    def _build_local_vol(self):
        """Compute Dupire local vol on the input implied-vol grid."""
        K, T  = self.strikes, self.maturities
        σ     = self.sigma_iv
        dT, dK = T[1] - T[0], K[1] - K[0]

        σ_T = _central_diff(σ, 0, dT)
        σ_K = _central_diff(σ, 1, dK)

        KK, TT = np.meshgrid(K, T)
        r = HestonParams.r  # risk-free rate is constant in the coursework

        # Dupire numerator & denominator
        num = (σ**2
               + 2 * TT * σ * σ_T
               + 2 * r * TT * KK * σ * σ_K
               + (KK**2) * (σ_K**2) * TT)

        base = 1 + KK * σ * σ_K
        base = np.where(np.abs(base) < 1e-12, 1e-12, base)  # avoid /0
        den  = base**2

        self.local_vol = np.sqrt(np.maximum(num / den, 1e-30))

        # Build QuantLib bilinear interpolator (expects zMatrix transposed)
        self._interp = ql.BilinearInterpolation(
            self.strikes.tolist(),           # x-axis (K)
            self.maturities.tolist(),        # y-axis (T)
            self.local_vol.T.tolist()        # transpose for QuantLib layout
        )

    # ---------------------------------------------------------------------
    #  public API
    # ---------------------------------------------------------------------
    def __call__(self, S: float, t: float) -> float:
        """Evaluate σ_loc(S,t) with t ≥ 1e-8 and box-clamped (S,t)."""
        S_c, t_c = self._clamp(S, max(t, 1e-8))
        return float(self._interp(S_c, t_c, True))

    # ---------------------------------------------------------------------
    #  plotting helpers
    # ---------------------------------------------------------------------
    def _plot_heatmap(self, vmin: float, vmax: float):
        X, Y = np.meshgrid(self.strikes, self.maturities)
        plt.figure(figsize=(8, 6))
        plt.pcolormesh(
            X,
            Y,
            self.local_vol,
            shading="auto",
            cmap="turbo",
            vmin=vmin,
            vmax=vmax,
        )
        plt.colorbar(label="Local vol")
        plt.xlabel("Strike")
        plt.ylabel("Maturity (years)")
        plt.title("Dupire local-vol surface (Task 4)")
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "task4_local_vol_surface.png", dpi=150)
        plt.close()

    def _plot_surface3d(self, vmin: float, vmax: float):
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – registers 3-D projection
        X, Y = np.meshgrid(self.strikes, self.maturities)
        fig  = plt.figure(figsize=(8, 6))
        ax   = fig.add_subplot(projection="3d")
        surf = ax.plot_surface(
            X,
            Y,
            self.local_vol,
            cmap="turbo",
            rstride=1,
            cstride=1,
            linewidth=0.25,        # thin black mesh
            edgecolor="k",
            antialiased=True,
            vmin=vmin,
            vmax=vmax,
        )
        fig.colorbar(surf, shrink=0.7, aspect=12, label="Local vol")
        ax.set_xlabel("Strike")
        ax.set_ylabel("Maturity (years)")
        ax.set_zlabel("Local vol")
        ax.set_title("Dupire local-vol surface (Task 4, 3-D)")
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "task4_local_vol_surface_3d.png", dpi=150)
        plt.close()

    # ---------------------------------------------------------------------
    def plot(
        self,
        vmin: float | None = None,
        vmax: float | None = None,
    ):
        """
        Save both 2-D heat-map and 3-D perspective plots using consistent,
        data-driven colour limits.

        If vmin / vmax are not supplied, the 1 %- and 99 %-percentiles of the
        local-vol matrix are used (robust to extreme outliers).
        """
        ensure_dir(PLOT_DIR)

        finite_vals = self.local_vol[np.isfinite(self.local_vol)]

        if vmin is None:
            vmin = np.nanpercentile(finite_vals, 1.0)
        if vmax is None:
            vmax = np.nanpercentile(finite_vals, 99.0)

        # make sure vmax > vmin by a usable margin
        if np.isclose(vmax, vmin):
            vmax = vmin + 1e-3

        self._plot_heatmap(vmin, vmax)
        self._plot_surface3d(vmin, vmax)

# ---------------------------------------------------------------------------
#  Monte-Carlo simulation under local-vol    (log-space Euler scheme)
# ---------------------------------------------------------------------------
def simulate_local_vol_paths(
    lv: DupireSurface,
    S0: float,
    r: float,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int = 1,
) -> np.ndarray:
    """
    Euler–Maruyama in **log space**  d ln S = (r−½σ²)dt + σ dW,
    which guarantees S_t > 0 and avoids overflow for large σ.
    """
    dt  = T / n_steps
    rng = np.random.default_rng(seed)

    lnS = np.full(n_paths, np.log(S0), dtype=float)   # start in log-space

    # warn once if extreme vols appear
    warn_raised = False

    for step in range(n_steps):
        t_curr  = (step + 1) * dt
        S_curr  = np.exp(lnS)                         # back to level
        sig_vec = np.fromiter((lv(s, t_curr) for s in S_curr), dtype=float)

        if not warn_raised and np.any(sig_vec > 5.0):
            warnings.warn(
                "Very high local vols (>500 %) encountered – "
                "log-Euler scheme keeps simulation stable.",
                RuntimeWarning,
            )
            warn_raised = True

        dW   = rng.standard_normal(n_paths) * np.sqrt(dt)
        lnS += (r - 0.5 * sig_vec**2) * dt + sig_vec * dW

    return np.exp(lnS)                                # back to level prices

# ---------------------------------------------------------------------------
#  Monte-Carlo pricing helper
# ---------------------------------------------------------------------------
def price_call_local_vol(
    lv: DupireSurface,
    S0: float,
    K: float,
    r: float,
    T: float,
    n_steps: int = 252,
    n_paths: int = 25_000,
) -> Tuple[float, float]:
    """Price a European call via local-vol MC; return (price, standard error)."""
    ST     = simulate_local_vol_paths(lv, S0, r, T, n_steps, n_paths)
    payoff = np.maximum(ST - K, 0.0)
    disc   = np.exp(-r * T)
    price  = disc * payoff.mean()
    se     = disc * payoff.std(ddof=1) / np.sqrt(n_paths)
    return price, se

# ---------------------------------------------------------------------------
#  Analytic Heston price at arbitrary spot   (cap integration order at 192)
# ---------------------------------------------------------------------------
def heston_price_given_spot(
    params: HestonParams,
    S_now: float,
    K: float,
    T: float,
) -> float:
    """Analytic Heston price with spot = S_now (integration order ≤ 192)."""
    tmp = HestonParams(
        S0=S_now,
        v0=params.v0,
        kappa=params.kappa,
        theta=params.theta,
        xi=params.xi,
        rho=params.rho,
        r=params.r,
        q=params.q,
    )
    model  = build_heston_model(tmp)
    engine = ql.AnalyticHestonEngine(model, 192)      # QuantLib hard limit
    payoff = ql.PlainVanillaPayoff(ql.Option.Call, K)
    option = ql.VanillaOption(payoff, ql.EuropeanExercise(date_from_years(T)))
    option.setPricingEngine(engine)
    price = option.NPV()

    # enforce arbitrage bounds at this spot
    intrinsic = max(
        S_now * np.exp(-params.q * T) - K * np.exp(-params.r * T),
        0.0,
    )
    price = max(price, intrinsic)
    price = min(price, S_now)
    return price

# ---------------------------------------------------------------------------
#  Driver – run Task 4
# ---------------------------------------------------------------------------
def run_task() -> None:
    ensure_dir(DATA_DIR)
    ensure_dir(PLOT_DIR)

    iv_csv = DATA_DIR / "heston_iv_surface.csv"
    if not iv_csv.exists():
        raise FileNotFoundError("IV surface CSV not found – run Task 3 first.")

    iv_df      = pd.read_csv(iv_csv, index_col=0)
    maturities = iv_df.index.astype(float).values
    strikes    = iv_df.columns.astype(float).values

    lv = DupireSurface(strikes, maturities, iv_df.values.astype(float))
    lv.plot()                                   # ← colour limits now automatic

    # quick validation price
    params     = HestonParams()
    K_test     = float(parameters.TEST_STRIKE)
    T_test     = float(parameters.TEST_MATURITY)
    price, se  = price_call_local_vol(lv, params.S0, K_test, params.r, T_test)
    print(
        f"[Task 4] Dupire price for K={K_test}, T={T_test:.2f}: "
        f"{price:.4f} ± {se:.4f} (1σ)"
    )

    return price, se

# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_task()
