# ---------------------------------------------------------------------------
#  clean_workspace.py   UPDATED (project root)
# ---------------------------------------------------------------------------

"""Utility script - wipe generated plots & data for a fresh run.

Run from the project root:

    python clean_workspace.py            # delete PNG + CSV outputs only
    python clean_workspace.py --all      # also remove empty folders, __pycache__, *.pyc, *~
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT      = Path(__file__).resolve().parent
PLOT_DIR  = ROOT / "plots"
DATA_DIR  = ROOT / "data"

EXTS_BASIC = (".png", ".csv")
EXTRA_PATTERNS = ["__pycache__", "*.pyc", "*~"]


def _delete_path(p: Path) -> None:
    """Delete file or directory *silently* (ignore missing / permission errors)."""
    try:
        if p.is_file():
            p.unlink(missing_ok=True)  # Python ≥3.8
        elif p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
    except PermissionError:
        # On Windows a file may be locked by another process (e.g. VS Code image preview).
        # We just skip it – user can close the handle and re‑run the cleaner.
        print(f"[warn] Could not remove {p} - file in use?")


def clean_workspace(all_files: bool = False) -> None:
    removed: list[Path] = []

    # basic outputs ---------------------------------------------------------
    for folder in (PLOT_DIR, DATA_DIR):
        if folder.exists():
            for item in folder.rglob("*"):
                if item.suffix in EXTS_BASIC:
                    _delete_path(item)
                    removed.append(item)
            # Optionally remove the now‑empty folder when --all is given
            if all_files and not any(folder.iterdir()):
                _delete_path(folder); removed.append(folder)

    # optional extra cleanup -----------------------------------------------
    if all_files:
        for pattern in EXTRA_PATTERNS:
            for item in ROOT.rglob(pattern):
                _delete_path(item); removed.append(item)

    if removed:
        print("Removed:")
        for p in removed:
            try:
                rel = p.relative_to(ROOT)
            except ValueError:
                rel = p
            print("  ", rel)
    else:
        print("Nothing to remove - workspace already clean or files were locked.")


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Clean generated data & plots.")
    ap.add_argument("--all", action="store_true", help="also remove empty folders, caches & backups")
    return ap.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    clean_workspace(all_files=args.all)

# ---------------------------------------------------------------------------
