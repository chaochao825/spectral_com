from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    src = str(root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    try:
        import pytest
    except ModuleNotFoundError:
        print('pytest is required; install the project with pip install -e ".[dev]"')
        return 2
    return int(pytest.main([str(root / "tests"), "-c", str(root / "pyproject.toml")]))


if __name__ == "__main__":
    raise SystemExit(main())
