"""run_test.py — locked test evaluator (§11). Refuses unless test_set_locked:true.

Mechanical protection preserved: lock check, no-overwrite without
--force --reason, audit log. Delegates the authorized evaluation to
evaluate_locked.py (frozen models/thresholds/localization, one-way gate).
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from evaluate_locked import main as locked_main
    return locked_main()


if __name__ == "__main__":
    sys.exit(main())
