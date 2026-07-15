from __future__ import annotations

import csv
import sys
from pathlib import Path


def count_csv_rows(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    root = Path(args[0] if args else "results")
    spectral = root / "metrics" / "spectral_metrics.csv"
    dynamic = root / "metrics" / "dynamic_metrics.csv"
    report = root / "report.md"
    eigenspectra = root / "plots" / "eigenspectra"
    heatmaps = root / "plots" / "heatmaps"
    eigenvalues = root / "eigenvalues"
    print(f"root={root}")
    print(f"spectral_metrics_exists={spectral.exists()}")
    print(f"spectral_rows={count_csv_rows(spectral)}")
    print(f"dynamic_metrics_exists={dynamic.exists()}")
    print(f"dynamic_rows={count_csv_rows(dynamic)}")
    print(f"report_exists={report.exists()}")
    print(f"eigenspectra_png={len(list(eigenspectra.glob('*.png'))) if eigenspectra.exists() else 0}")
    print(f"heatmap_png={len(list(heatmaps.glob('*.png'))) if heatmaps.exists() else 0}")
    print(f"eigen_json={len(list(eigenvalues.glob('*.json'))) if eigenvalues.exists() else 0}")
    if report.exists():
        print("report_head:")
        print("\n".join(report.read_text(encoding="utf-8").splitlines()[:6]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

