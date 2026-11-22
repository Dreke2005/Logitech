"""Launch the optional paper-reading workbench from the main project folder."""

# Author: Victor Lancieux
# Date: 2026-11-22 (paper workbench integrated into the main launcher)
# Purpose: Keep the friend's paper-reading app available through the root uv environment.

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
FRIEND_DIR = PROJECT_DIR / "Logitech-Keypad"
sys.path.insert(0, str(FRIEND_DIR))

from paper_workbench import main  # noqa: E402


if __name__ == "__main__":
    main()
