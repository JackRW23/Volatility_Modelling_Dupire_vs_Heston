# ---------------------------------------------------------------------------
#  src/vol_modelling/parameters.py
# ---------------------------------------------------------------------------

"""Centralized parameters configuration for the Heston model and tasks."""

from vol_modelling.common import HestonParams

# Default model parameters (from coursework specification)
current_params = HestonParams()  # Global instance of HestonParams with default values

# Default strike and maturity grid settings
STRIKE_MIN = 175.0
STRIKE_MAX = 270.0
MATURITY_MIN = 0.05
MATURITY_MAX = 2.0
GRID_SIZE = 50  # number of points for strike and maturity grids

# Default test option (off the grid) parameters (used in Task 4 & 5)
TEST_STRIKE = 221.0
TEST_MATURITY = 2.0
# Snapshot times for Task 5 (years before maturity)
# (These values represent times before TEST_MATURITY at which to reprice)
SNAPSHOT_TIMES = [0.20, 0.60, 1.00, 1.40, 1.80]

# Monte Carlo paths for Task 5 repricing (Dupire local vol simulation)
MC_PATHS_TASK5 = 3000

def override_heston_params(S0=None, v0=None, kappa=None, theta=None, xi=None, rho=None, r=None, q=None):
    """Override global Heston model parameters with new values where provided."""
    global current_params
    curr = current_params
    # Create a new HestonParams with overrides (use existing values if not provided)
    current_params = HestonParams(
        S0 if S0 is not None else curr.S0,
        v0 if v0 is not None else curr.v0,
        kappa if kappa is not None else curr.kappa,
        theta if theta is not None else curr.theta,
        xi if xi is not None else curr.xi,
        rho if rho is not None else curr.rho,
        r if r is not None else curr.r,
        q if q is not None else curr.q
    )

def override_strike_bounds(min_val=None, max_val=None):
    """Override strike range bounds."""
    global STRIKE_MIN, STRIKE_MAX
    if min_val is not None:
        STRIKE_MIN = float(min_val)
    if max_val is not None:
        STRIKE_MAX = float(max_val)

def override_maturity_bounds(min_val=None, max_val=None):
    """Override maturity range bounds."""
    global MATURITY_MIN, MATURITY_MAX
    if min_val is not None:
        MATURITY_MIN = float(min_val)
    if max_val is not None:
        MATURITY_MAX = float(max_val)

def override_grid_size(n=None):
    """Override the grid size (number of strike/maturity points)."""
    global GRID_SIZE
    if n is not None:
        GRID_SIZE = int(n)

def override_test_option(strike=None, maturity=None):
    """Override the off-grid test option strike and maturity (Task 4 & 5)."""
    global TEST_STRIKE, TEST_MATURITY
    if strike is not None:
        TEST_STRIKE = float(strike)
    if maturity is not None:
        TEST_MATURITY = float(maturity)

def override_mc_paths_task5(n=None):
    """Override the number of Monte Carlo paths for Task 5 pricing simulation."""
    global MC_PATHS_TASK5
    if n is not None:
        MC_PATHS_TASK5 = int(n)
