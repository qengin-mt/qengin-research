# QEngin MT research reproduction

This repository is the project's current data-availability package. It contains the research datasets and analysis scripts needed to regenerate the reported statistical tables and figures for *From Translation Errors to Quality Scores: ISO 5060-Guided LLM Evaluation with Human Validation*. The repository will be archived in a permanent research data repository when one is selected.

## Research history

The study was conducted over an extended period in a separate working repository. This repository was assembled afterward for data availability and reproducibility. It is not the original development or model-inference repository. Its contents will be transferred to a permanent research data archive later.

## Reproduction boundary

The reproducible workflow starts from the preserved model response JSON, human validity decisions, semantic-link decisions, model-to-model pair judgments, original human-description corpus, and saved COMETKiwi segment scores. It does not rerun historical LLM inference or recreate human judgments. Exact historical LLM prompts, immutable model revisions, and complete generation settings were not retained. The COMETKiwi scores used in the paper are included as an analysis input; model inference is not needed to reproduce the reported comparisons. An optional runner is provided to generate new segment-level scores with the WMT22, WMT23-XL, or WMT23-XXL COMETKiwi checkpoint.

The four Luna segment-88 findings were recovered from Luna GPT after the API call failed. They are represented in the final research tables and model-pair workbooks; the saved Luna JSON has no response for that segment. The 134-segment saved-output comparison and separate 135-segment completed-table analysis are both reproduced.

## Contents

- datasets/nmt/: only research inputs consumed by the analysis, plus the source corpus description. Gemma's final-label tables are normalized to true TSV and omit reviewer submission IDs, respondent IDs, timestamps, and free-text reviewer metadata not used by the analysis.
- scripts/: analysis scripts that regenerate statistics, tables, and figures from the included datasets.
- outputs/: generated analysis products (created by the scripts and excluded from version control).
- pyproject.toml, uv.lock, .python-version: pinned Python 3.12 analysis environment managed with uv.

The manuscript and submission files are kept separately from this dataset-and-scripts repository. Archived experiments, intermediate survey exports, unrelated QE-engine code, API credentials, local virtual environments, and downloaded related-work PDFs are intentionally excluded. COMET inference uses a separate `.venv-comet` environment because its dependency constraints differ from the analysis environment.

## Set up the Python environment

Install uv using the official installation instructions for your operating system, then from this repository root run:

    uv python install 3.12
    uv sync --locked

This creates the project environment in .venv/. Run project Python commands with uv run; do not create or copy a virtual environment into the repository.

## Recompute analyses and tables

Run from the repository root:

    uv run python scripts/analyze.py --bootstrap 10000 --seed 20261012
    uv run python scripts/analyze_cross_evaluator_pairs.py
    uv run python scripts/render_scatter.py

These commands regenerate analysis JSON, TSV and LaTeX tables, and the scatter plot under `outputs/`. They make no network calls and do not modify source data. The scripts check segment alignment, model-prediction traceability, validity labels, pair decisions, and required Luna segment-88 findings.

## Optional: generate COMETKiwi segment scores

The scorer accepts a JSON array or TSV with source and target text. For the included corpus, use `datasets/nmt/human_gold/anotovane_chyby_NMT.json`; it contains the 135 source/translation pairs. COMET checkpoints are downloaded from Hugging Face on first use. Accept any checkpoint-specific license terms on its model page before downloading; WMT23 XL/XXL use a non-commercial license. If Hugging Face requires authentication, log in from the COMET environment with:

    .venv-comet/bin/hf auth login

Create an isolated inference environment with uv:

    uv python install 3.12
    uv venv --python 3.12 .venv-comet
    uv pip install --python .venv-comet/bin/python -r comet-requirements.txt

Score the full 135-segment corpus. The XXL checkpoint is memory intensive; reduce `--batch-size` to `1` if needed, or set `--gpus 0` to run on CPU:

    .venv-comet/bin/python scripts/score_cometkiwi.py --model wmt22-da --input datasets/nmt/human_gold/anotovane_chyby_NMT.json --output-dir outputs/cometkiwi --batch-size 4 --gpus 1
    .venv-comet/bin/python scripts/score_cometkiwi.py --model wmt23-xl --input datasets/nmt/human_gold/anotovane_chyby_NMT.json --output-dir outputs/cometkiwi --batch-size 1 --gpus 1
    .venv-comet/bin/python scripts/score_cometkiwi.py --model wmt23-xxl --input datasets/nmt/human_gold/anotovane_chyby_NMT.json --output-dir outputs/cometkiwi --batch-size 1 --gpus 1

Each run writes `cometkiwi_<model>.tsv` with source, target, and one segment score. Existing outputs are protected; add `--overwrite` only when intentionally replacing a score file. To regenerate the paper's analysis with a selected checkpoint's scores, run the analysis environment's Python command, for example:

    uv run python scripts/analyze.py --cometkiwi-scores outputs/cometkiwi/cometkiwi_wmt23-xl.tsv --bootstrap 10000 --seed 20261012

Use the corresponding score file in `outputs/cometkiwi/` to analyze another checkpoint. This writes the usual analysis products under `outputs/`, replacing the prior generated analysis products there.

To reproduce the manuscript's joint checkpoint comparison without overwriting the default analysis artifacts, run:

    uv run python scripts/analyze_cometkiwi_checkpoints.py --bootstrap 10000 --seed 20261012

This runs the analysis separately for WMT22 DA, WMT23 DA-XL, and WMT23 DA-XXL, writes checkpoint-specific artifacts below `outputs/cometkiwi-analysis/`, and creates the manuscript table and its machine-readable summary under `paper/generated/`.

## Included analysis inputs

| Input | Use |
|---|---|
| datasets/nmt/human_gold/anotovane_chyby_NMT.json | Canonical segments and original human-description denominator |
| datasets/nmt/{gemma-4-31b-qat,sol,luna}/model_results.json | Saved LLM predictions and traceability checks |
| final_labels.tsv and final_accepted.tsv for each model | Final validity decisions and complete/accepted error populations |
| accepted_human_gold_pairs*.tsv | Human semantic-link decisions |
| datasets/nmt/FINAL *error_pairs.xlsx | Direct reviewed model-to-model pair decisions |
| datasets/nmt/cometkiwi_and_qe_scores.tsv | Unified segment table with WMT22 DA, WMT23 DA-XL, WMT23 DA-XXL, and Gemma/Sol/Luna error-derived quality scores |

`datasets/nmt/cometkiwi_and_qe_scores.tsv` combines all three checkpoint scores with the error-derived score columns from the original WMT22 comparison table. Source and target pairs were verified before joining. The original WMT22 comparison TSV remains as the default input to `analyze.py`. Checkpoint score outputs used by the joint runner are in `outputs/cometkiwi/`. `scripts/score_cometkiwi.py` can create new COMETKiwi score TSVs from the canonical source/target corpus.

The error-derived score is calculated from the included complete error-annotation sets using target whitespace-token count, severity penalties neutral/minor/major/critical = 0/0.25/0.5/1, and Q = 1/(1+P/V). For each segment, the script sums severity penalties to obtain P, divides by the target whitespace-token count V to obtain E=P/V, and computes Q=1/(1+E).

## Archival note

This GitHub repository currently provides access to the research materials and reproduction code. It is intended as an interim location. The project plans to deposit an archived version in a permanent research data repository later.
