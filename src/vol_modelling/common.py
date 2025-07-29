from __future__ import annotations
import warnings
from dataclasses import dataclass
from typing import Tuple
from pathlib import Path
import numpy as np
import pandas as pd
import QuantLib as ql

# ---------------------------------------------------------------------------
#  Global constants & date helpers
# ---------------------------------------------------------------------------

TODAY = ql.Date.todaysDate()
ql.Settings.instance().evaluationDate = TODAY
DAY_COUNT = ql.Actual365Fixed()
CAL = ql.NullCalendar()

def date_from_years(years: float) -> ql.Date:
    """Convert *fractional* years into a QuantLib calendar date."""
    return CAL.advance(TODAY, ql.Period(int(years * 365), ql.Days))

# ---------------------------------------------------------------------------
#  Parameter container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HestonParams:
    """Immutable container for Heston model parameters.
    
    Attributes:
        S0 (float): Initial asset price.
        v0 (float): Initial variance.
        kappa (float): Mean reversion rate of variance.
        theta (float): Long-run mean variance (target level).
        xi (float): Volatility of volatility (standard deviation of variance).
        rho (float): Correlation between asset and variance Brownian motions.
        r (float): Risk-free interest rate.
        q (float): Dividend yield.
    
    Default values are set by coursework task 1.
    """
    S0: float = 200.0
    v0: float = 0.04
    kappa: float = 1.5
    theta: float = 0.04
    xi: float = 0.3
    rho: float = -0.7
    r: float = 0.05
    q: float = 0.0

# ---------------------------------------------------------------------------
#  Builders
# ---------------------------------------------------------------------------

def build_heston_process(p: HestonParams) -> ql.HestonProcess:
    """Create a QuantLib HestonProcess for the given model parameters.
    
    The HestonProcess is defined by flat term-structure assumptions for the 
    risk-free rate and dividend yield, the initial spot price, and the Heston 
    parameters (initial variance, mean reversion speed and level, volatility 
    of variance, and correlation).
    
    Args:
        p: HestonParams object containing model parameters.
    Returns:
        ql.HestonProcess: QuantLib stochastic process for Heston dynamics.
    """
    r_ts = ql.YieldTermStructureHandle(ql.FlatForward(TODAY, p.r, DAY_COUNT))
    q_ts = ql.YieldTermStructureHandle(ql.FlatForward(TODAY, p.q, DAY_COUNT))
    spot = ql.QuoteHandle(ql.SimpleQuote(p.S0))
    return ql.HestonProcess(r_ts, q_ts, spot, p.v0, p.kappa, p.theta, p.xi, p.rho)

def build_heston_model(p: HestonParams) -> ql.HestonModel:
    """Create a QuantLib HestonModel from the given parameters.
    
    This wraps a HestonProcess (created via build_heston_process) into a 
    HestonModel object, which is used for analytical option pricing (closed-form).
    
    Args:
        p: HestonParams object with model parameters.
    Returns:
        ql.HestonModel: QuantLib model object for Heston.
    """
    return ql.HestonModel(build_heston_process(p))

def build_bs_process(p: HestonParams, vol: float = 0.25) -> ql.BlackScholesMertonProcess:
    """Construct a Black-Scholes-Merton process with given parameters and volatility.
    
    Uses the same spot price, risk-free rate (r), and dividend yield (q) from 
    the provided HestonParams `p`, and a flat volatility term structure with 
    constant volatility `vol`. This is typically used for implied volatility calculations.
    
    Args:
        p: HestonParams for base values (S0, r, q).
        vol: Constant volatility to use (default 0.25, i.e., 25%).
    Returns:
        ql.BlackScholesMertonProcess: QuantLib process for Black-Scholes dynamics.
    """
    spot = ql.QuoteHandle(ql.SimpleQuote(p.S0))
    r_ts = ql.YieldTermStructureHandle(ql.FlatForward(TODAY, p.r, DAY_COUNT))
    q_ts = ql.YieldTermStructureHandle(ql.FlatForward(TODAY, p.q, DAY_COUNT))
    vol_ts = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(TODAY, CAL, vol, DAY_COUNT))
    return ql.BlackScholesMertonProcess(spot, q_ts, r_ts, vol_ts)

# ---------------------------------------------------------------------------
#  Task 1 helper - Monte-Carlo simulation
# ---------------------------------------------------------------------------

def _make_gaussian_sequence_generator(dim: int) -> ql.GaussianRandomSequenceGenerator:
    """Internal helper to create a Gaussian random sequence generator of a given dimension.
    
    This uses QuantLib's UniformRandomSequenceGenerator to generate `dim` uniform 
    random numbers, and then converts them to Gaussian (normal) random numbers.
    Returns a GaussianRandomSequenceGenerator that can be used for multi-dimensional path generation.
    
    Args:
        dim: The total number of Gaussian random variates needed.
    Returns:
        ql.GaussianRandomSequenceGenerator: Generator producing sequences of Gaussian random numbers.
    """
    ur_gen = ql.UniformRandomGenerator()
    usg = ql.UniformRandomSequenceGenerator(dim, ur_gen)
    return ql.GaussianRandomSequenceGenerator(usg)

def simulate_heston_paths(
    process: ql.HestonProcess,
    maturity: float,
    n_steps: int,
    n_paths: int,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate sample paths for the Heston model using QuantLib's path generator.
    
    Generates `n_paths` paths for the asset price and variance processes over 
    `n_steps` time steps up to time `maturity`. Uses QuantLib's GaussianMultiPathGenerator 
    for the HestonProcess.
    
    Args:
        process: The QuantLib HestonProcess to simulate.
        maturity: Time horizon (in years) to simulate up to.
        n_steps: Number of time steps in the simulation (discretization of [0, maturity]).
        n_paths: Number of independent paths to generate.
        seed: Random seed for simulation (not explicitly used in this implementation).
    Returns:
        tuple(np.ndarray, np.ndarray, np.ndarray):
            - t: 1D array of time points (length n_steps+1).
            - S: 2D array of shape (n_paths, n_steps+1) of simulated asset prices.
            - V: 2D array of shape (n_paths, n_steps+1) of simulated variance values.
    
    Note:
        The QuantLib random number generator is not directly seeded here (it uses a default seed for reproducibility). 
        Each path consists of `n_steps` intervals (n_steps+1 points including time 0).
    """
    time_grid = ql.TimeGrid(maturity, n_steps)
    dim = int(process.factors()) * n_steps
    gsg = _make_gaussian_sequence_generator(dim)
    generator = ql.GaussianMultiPathGenerator(process, time_grid, gsg, False)
    S = np.empty((n_paths, len(time_grid)))
    V = np.empty_like(S)
    for i in range(n_paths):
        path = generator.next().value()
        S[i, :] = [path[0][j] for j in range(len(time_grid))]
        V[i, :] = [path[1][j] for j in range(len(time_grid))]
    t = np.array([time_grid[i] for i in range(len(time_grid))], dtype=float)
    return t, S, V

# ---------------------------------------------------------------------------
#  Task 2 helper – Analytic Heston pricing & price surface
# ---------------------------------------------------------------------------

def heston_call_price(
    model: ql.HestonModel,
    maturity: float,
    strike: float,
    cache: dict,
    integration_order: int = 192,
) -> float:
    """
    Compute a European call option price under the Heston model.
    
    This uses QuantLib's AnalyticHestonEngine for pricing. To improve efficiency 
    when pricing multiple options with the same maturity, a cache of engines is 
    used (keyed by maturity). A high integration order (default 192) is specified 
    for numerical integration to enhance stability/accuracy of the Heston pricing.
    
    Args:
        model: QuantLib HestonModel for the desired parameters.
        maturity: Time to expiration in years.
        strike: Strike price of the call option.
        cache: Dictionary for caching AnalyticHestonEngine instances per maturity.
        integration_order: Integration points for numerical Fourier inversion (default 192).
    Returns:
        float: The model price of the European call option.
    """
    key = f"T={maturity:.6f}"
    if key not in cache:
        cache[key] = ql.AnalyticHestonEngine(model, integration_order)
    payoff   = ql.PlainVanillaPayoff(ql.Option.Call, strike)
    exercise = ql.EuropeanExercise(date_from_years(maturity))
    option   = ql.VanillaOption(payoff, exercise)
    option.setPricingEngine(cache[key])
    price = option.NPV()
    # Enforce arbitrage bounds: price >= intrinsic value and >= 0
    if price < 0.0:
        price = 0.0
    return price

def build_price_surface(params: HestonParams, maturities: np.ndarray, strikes: np.ndarray) -> pd.DataFrame:
    """Build a Heston call price surface on the given maturity/strike grid.
    
    Computes the European call option price for each combination of maturity in 
    `maturities` and strike in `strikes` under the Heston model specified by `params`. 
    Returns the results as a pandas DataFrame (maturities as index, strikes as columns).
    The pricing uses the analytic Heston formula and enforces basic no-arbitrage bounds 
    (prices are floored at intrinsic value and capped at S0).
    
    Args:
        params: HestonParams object defining the model (including S0, r, q, etc.).
        maturities: 1D array of maturity times (years) for the surface.
        strikes: 1D array of strike prices for the surface.
    Returns:
        pd.DataFrame: DataFrame of call prices with index = maturities and columns = strikes.
    """
    model = build_heston_model(params)
    cache: dict[str, ql.AnalyticHestonEngine] = {}
    surf = np.empty((len(maturities), len(strikes)))
    for i, T in enumerate(maturities):
        for j, K in enumerate(strikes):
            price = heston_call_price(model, float(T), float(K), cache)
            intrinsic = max(params.S0 * np.exp(-params.q * T) - K * np.exp(-params.r * T), 0.0)
            if price < intrinsic:
                price = intrinsic
            if price > params.S0:
                price = params.S0
            surf[i, j] = price
    df = pd.DataFrame(surf, index=maturities, columns=strikes)
    df.index.name = "Maturity"; df.columns.name = "Strike"
    return df

# ---------------------------------------------------------------------------
#  Task 3 helper – implied volatility inversion & surface
# ---------------------------------------------------------------------------

def implied_volatility(
    price: float,
    strike: float,
    maturity: float,
    params: HestonParams,
    vol_guess: float = 0.25,
    tol: float = 1e-5,
    max_evals: int = 10000,
    min_vol: float = 1e-12,
    max_vol: float = 5.0,
) -> float:
    """Compute implied volatility from a call price via QuantLib's solver, with robust bounds.
    
    Finds the implied Black-Scholes volatility that would produce the given `price` 
    for a call option with the specified strike and maturity, under the market 
    conditions in `params` (for underlying price, r, q). The search is performed 
    using QuantLib's `VanillaOption.impliedVolatility` solver.
    
    Various adjustments are made for numerical robustness:
      - The input price is constrained to be at least a tiny positive value (to avoid solver issues with 0).
      - The price is floored at intrinsic value and capped at the forward underlying price (no-arbitrage bounds).
      - The solver is given an initial guess `vol_guess` and allowed to search between `min_vol` and `max_vol`.
      - If the first attempt fails to converge, the max volatility bound is doubled and the solver retried.
      - If it still fails and the price is essentially intrinsic, 0.0 is returned (implied vol ~ 0). Otherwise, `max_vol` is returned.
    
    Args:
        price: The call option price.
        strike: Strike price of the option.
        maturity: Time to maturity in years.
        params: HestonParams for the underlying (provides S0, r, q for forward price).
        vol_guess: Initial guess for volatility (default 0.25 or 25%).
        tol: Solver tolerance for convergence (default 1e-5).
        max_evals: Maximum solver iterations (default 10000).
        min_vol: Lower bound for volatility (default 1e-12, effectively 0).
        max_vol: Upper bound for volatility (default 5.0, i.e., 500%).
    Returns:
        float: Implied volatility (in fractional form, e.g., 0.2 for 20%).
    """
    payoff   = ql.PlainVanillaPayoff(ql.Option.Call, strike)
    exercise = ql.EuropeanExercise(date_from_years(maturity))
    option   = ql.VanillaOption(payoff, exercise)
    tiny = 1e-10
    if price <= 0.0:
        price = tiny
    elif 0.0 < price < tiny:
        price = tiny
    fwd_intrinsic = params.S0 * np.exp(-params.q * maturity) - strike * np.exp(-params.r * maturity)
    intrinsic = fwd_intrinsic if fwd_intrinsic > 0.0 else 0.0
    if price < intrinsic:
        price = intrinsic
    cap = params.S0 * np.exp(-params.q * maturity)
    if price > cap:
        price = cap
    bs_proc = build_bs_process(params, vol_guess)
    try:
        iv = option.impliedVolatility(price, bs_proc, tol, max_evals, min_vol, max_vol)
    except RuntimeError:
        try:
            iv = option.impliedVolatility(price, bs_proc, tol, max_evals, min_vol, max_vol * 2)
        except RuntimeError:
            if price <= intrinsic + 1e-16:
                return 0.0
            iv = max_vol
    return iv

def build_iv_surface(price_df: pd.DataFrame, params: HestonParams) -> pd.DataFrame:
    """Compute the implied-volatility surface from a Heston call price grid.
    
    For each call price in the input DataFrame (indexed by maturity, with strike columns), 
    this function computes the implied volatility. Returns a DataFrame of the same shape 
    with the implied volatilities.
    
    Args:
        price_df: DataFrame of call option prices (index = maturities, columns = strikes).
        params: HestonParams used for those prices (providing S0, r, q for forward price calculation).
    Returns:
        pd.DataFrame: Implied volatility surface, with index = maturities and columns = strikes.
    """
    maturities = price_df.index.astype(float).values
    strikes = price_df.columns.astype(float).values
    iv_matrix = np.empty(price_df.shape)
    for i, T in enumerate(maturities):
        for j, K in enumerate(strikes):
            price = float(price_df.iloc[i, j])
            iv = implied_volatility(price, float(K), float(T), params)
            iv_matrix[i, j] = iv
    iv_df = pd.DataFrame(iv_matrix, index=maturities, columns=strikes)
    iv_df.index.name = "Maturity"; iv_df.columns.name = "Strike"
    return iv_df

# ---------------------------------------------------------------------------
#  Utility – ensure directory exists
# ---------------------------------------------------------------------------

def ensure_dir(path: Path) -> None:
    """Create directory if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)
