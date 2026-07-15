from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DPI = 300
COLORS = {"numpy_batched": "#4C78A8", "torch_cuda_batched": "#E45756"}


def setup_style() -> None:
    matplotlib.rcParams.update(
        {
            "font.size": 10,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "savefig.dpi": DPI,
            "savefig.bbox": "tight",
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot structured CUDA benchmark results.")
    parser.add_argument("--result-dir", default="results/cuda_benchmark_20260610")
    args = parser.parse_args()
    result_dir = Path(args.result_dir)
    out_dir = result_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    up = pd.read_csv(result_dir / "qwen_mlp_up.csv")
    down = pd.read_csv(result_dir / "qwen_mlp_down.csv")
    timing = pd.read_csv(result_dir / "phase1_timing.csv")
    setup_style()

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    labels = ["up/gate\n8960x1536", "down\n1536x8960"]
    x = np.arange(len(labels))
    width = 0.34
    for offset, backend in ((-width / 2, "numpy_batched"), (width / 2, "torch_cuda_batched")):
        values = []
        for frame in (up, down):
            values.append(float(frame[(frame["operation"] == "monarch_like") & (frame["backend"] == backend)]["seconds"].iloc[0]))
        axes[0].bar(x + offset, values, width, label=backend.replace("_", " "), color=COLORS[backend])
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Wall time (s)")
    axes[0].set_xlabel("Qwen MLP matrix")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=8)

    batch = timing.copy()
    batch["batch_size"] = batch["backend"].str.extract(r"(\d+)").astype(int)
    batch = batch.sort_values("batch_size")
    axes[1].plot(batch["batch_size"], batch["seconds"], marker="o", linewidth=1.7, color="#54A24B")
    axes[1].set_xticks(batch["batch_size"])
    axes[1].set_xlabel("Residual SVD batch size")
    axes[1].set_ylabel("Phase 1 down_proj wall time (s)")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "structured_cuda_benchmark.pdf", format="pdf")
    fig.savefig(out_dir / "structured_cuda_benchmark.png", format="png")
    plt.close(fig)


if __name__ == "__main__":
    main()
