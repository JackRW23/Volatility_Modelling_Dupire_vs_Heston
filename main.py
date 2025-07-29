# ---------------------------------------------------------------------------
#  main.py  (project root)
# ---------------------------------------------------------------------------

"""Master runner for Vol-Modelling coursework tasks."""

from __future__ import annotations

import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC  = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vol_modelling import task1_heston, task2_surface, task3_iv, task4_dupire, task5_reprice  # type: ignore
import vol_modelling.parameters as parameters  # (Param Centralization) import global parameters

# (Param Centralization) use default MC paths from global parameters
_DEF_MC_PATHS = parameters.MC_PATHS_TASK5

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run Vol-Modelling coursework pipeline")
    ap.add_argument("--mc-paths", type=int, default=_DEF_MC_PATHS,
                    help="Monte-Carlo paths for Task 5 (default 30,000)")
    return ap.parse_args()

# ---------------------------------------------------------------------------
#  Pipeline runner
# ---------------------------------------------------------------------------

def _run_pipeline():
    tasks = [
        ("Task 1 — Heston path simulation", task1_heston.run_task, {}),
        ("Task 2 — price surface",         task2_surface.run_task, {}),
        ("Task 3 — implied-vol surface",   task3_iv.run_task,       {}),
        ("Task 4 — Dupire local-vol",      task4_dupire.run_task,   {}),
        ("Task 5 — repricing comparison",  task5_reprice.run_task,  {}),
    ]
    for title, fn, kwargs in tasks:
        print(f"\n===== {title} =====")
        fn(**kwargs)

def main():
    args = _parse_args()
    # (Param Centralization) Override Task 5 Monte Carlo paths if provided via CLI
    parameters.override_mc_paths_task5(args.mc_paths)
    _run_pipeline()

if __name__ == "__main__":
    main()
