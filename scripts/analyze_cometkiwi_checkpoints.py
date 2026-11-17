#!/usr/bin/env python3
"""Run the error-score analysis against all stored COMETKiwi checkpoints."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = (
    ("WMT22 DA", "wmt22-da", ROOT / "outputs/cometkiwi/cometkiwi_wmt22-da.tsv"),
    ("WMT23 DA-XL", "wmt23-xl", ROOT / "outputs/cometkiwi/cometkiwi_wmt23-xl.tsv"),
    ("WMT23 DA-XXL", "wmt23-xxl", ROOT / "outputs/cometkiwi/cometkiwi_wmt23-xxl.tsv"),
)
MODELS = ("gemma", "sol", "luna")


def fmt_ci(values: list[float]) -> str:
    return f"[{values[0]:.3f}, {values[1]:.3f}]"


def latex_table(results: list[dict]) -> str:
    lines = [
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"COMETKiwi checkpoint & Error annotator & $N$ & Pearson $r$ [95\% CI] & Spearman $\rho$ [95\% CI] \\",
        r"\midrule",
    ]
    for result in results:
        for model in MODELS:
            score = result["scores"][model]
            lines.append(
                f'{result["label"]} & {model.title()} & {score["n"]} & '
                f'{score["pearson"]:.3f} {fmt_ci(score["pearson_ci95"])} & '
                f'{score["spearman"]:.3f} {fmt_ci(score["spearman_ci95"])} \\\\'
            )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument("--analysis-output-root", type=Path,
                        default=ROOT / "outputs/cometkiwi-analysis")
    parser.add_argument("--paper-table", type=Path,
                        default=ROOT / "paper/generated/cometkiwi_checkpoints.tex")
    parser.add_argument("--summary-json", type=Path,
                        default=ROOT / "paper/generated/cometkiwi_checkpoints.json")
    args = parser.parse_args()

    results = []
    for label, slug, scores_path in CHECKPOINTS:
        if not scores_path.is_file():
            raise FileNotFoundError(f"Missing checkpoint scores: {scores_path}")
        output_dir = args.analysis_output_root / slug
        command = [
            sys.executable, str(ROOT / "scripts/analyze.py"),
            "--cometkiwi-scores", str(scores_path),
            "--output-dir", str(output_dir),
            "--bootstrap", str(args.bootstrap),
            "--seed", str(args.seed),
        ]
        subprocess.run(command, check=True)
        analysis = json.loads((output_dir / "analysis.json").read_text(encoding="utf-8"))
        results.append({
            "label": label,
            "slug": slug,
            "score_file": str(scores_path.relative_to(ROOT)),
            "scores": analysis["scores"],
            "paired_pearson_differences": analysis["paired_pearson_differences"],
        })

    args.paper_table.parent.mkdir(parents=True, exist_ok=True)
    args.paper_table.write_text(latex_table(results), encoding="utf-8")
    args.summary_json.write_text(json.dumps({
        "bootstrap_replicates": args.bootstrap,
        "seed": args.seed,
        "checkpoints": results,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.paper_table}")
    print(f"Wrote {args.summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
