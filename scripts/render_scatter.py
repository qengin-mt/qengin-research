"""Render Figure 2 from archived segment scores without recomputing analyses."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    directory = Path(__file__).resolve().parents[1] / "outputs"
    with (directory / "segment_scores.tsv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.6), sharex=True, sharey=True,
                             layout="constrained")
    for ax, model, title in zip(axes, ("gemma", "sol", "luna"),
                                ("Gemma", "Sol", "Luna")):
        quality = [float(row[f"{model}_Q"]) for row in rows]
        kiwi = [float(row["cometkiwi"]) for row in rows]
        ax.scatter(kiwi, quality, s=22, alpha=0.65, edgecolors="none")
        ax.axhline(1, color="0.5", linestyle=":", linewidth=0.8)
        ax.set_title(f"{title}\nQ = 1: {sum(q == 1 for q in quality)} segments",
                     fontsize=12)
        ax.set_xlabel("COMETKiwi score", fontsize=11)
        ax.tick_params(axis="both", labelsize=10)
        ax.set_yticks([0.88, 0.92, 0.96, 1.00])
        ax.grid(alpha=0.2)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Quality Q from complete error-annotation set", fontsize=11)
    fig.savefig(directory / "score_scatter.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
