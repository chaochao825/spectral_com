from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = str(ROOT / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from llm_spectral_dynamics.publish_checks import validate_tree  # noqa: E402


def main() -> int:
    checked, errors = validate_tree(ROOT)
    if errors:
        for error in errors:
            print(f"ERROR {error}")
        print(f"publication check failed: {len(errors)} issue(s) across {checked} files")
        return 1
    print(f"publication check passed: {checked} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
