#!/usr/bin/env python3
"""Score source/translation TSV or JSON records with a COMETKiwi checkpoint."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

MODELS = {
    "wmt22-da": "Unbabel/wmt22-cometkiwi-da",
    "wmt23-xl": "Unbabel/wmt23-cometkiwi-da-xl",
    "wmt23-xxl": "Unbabel/wmt23-cometkiwi-da-xxl",
}


def records_from_file(path: Path, source_column: str | None, target_column: str | None):
    if path.suffix.lower() == ".json":
        records = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise ValueError("JSON input must be an array of objects")
        if not records:
            raise ValueError("Input contains no records")
        source_column = source_column or "source_sentence"
        target_column = target_column or "translation"
    elif path.suffix.lower() in {".tsv", ".txt"}:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream, delimiter="\t")
            if not reader.fieldnames:
                raise ValueError("TSV input must have a header row")
            source_column = source_column or next(
                (c for c in ("source_sentence", "source", "src") if c in reader.fieldnames), None
            )
            target_column = target_column or next(
                (c for c in ("target_sentence", "translation", "target", "mt") if c in reader.fieldnames), None
            )
            if not source_column or not target_column:
                raise ValueError(
                    "Could not identify source/target columns; set --source-column and --target-column"
                )
            records = list(reader)
    else:
        raise ValueError("Input must be a .json or tab-separated .tsv file")

    pairs = []
    for number, record in enumerate(records, 1):
        if not isinstance(record, dict):
            raise ValueError(f"Record {number} is not an object")
        source = record.get(source_column)
        target = record.get(target_column)
        if not isinstance(source, str) or not source.strip():
            raise ValueError(f"Record {number} has no non-empty {source_column!r}")
        if not isinstance(target, str) or not target.strip():
            raise ValueError(f"Record {number} has no non-empty {target_column!r}")
        pairs.append((source, target))
    if not pairs:
        raise ValueError("Input contains no records")
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True,
                        help="wmt22-da, wmt23-xl, or wmt23-xxl")
    parser.add_argument("--input", type=Path, required=True, help="JSON array or TSV with source/target columns")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-column", help="Override the source column name")
    parser.add_argument("--target-column", help="Override the target column name")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs; use 0 for CPU")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output file")
    args = parser.parse_args()
    if args.batch_size < 1 or args.gpus < 0:
        parser.error("--batch-size must be positive and --gpus must be zero or greater")
    if not args.input.is_file():
        parser.error(f"Input file does not exist: {args.input}")

    output_path = args.output_dir / f"cometkiwi_{args.model}.tsv"
    if output_path.exists() and not args.overwrite:
        parser.error(f"Output already exists: {output_path} (pass --overwrite to replace it)")
    pairs = records_from_file(args.input, args.source_column, args.target_column)

    try:
        from comet import download_model, load_from_checkpoint
    except ImportError as exc:
        raise SystemExit(
            "COMET is not installed in this Python environment. Follow the COMET setup commands in README.md."
        ) from exc

    model_id = MODELS[args.model]
    print(f"Loading {model_id}; scoring {len(pairs)} segments", file=sys.stderr, flush=True)
    model_path = download_model(model_id)
    model = load_from_checkpoint(model_path)
    predictions = model.predict(
        [{"src": source, "mt": target} for source, target in pairs],
        batch_size=args.batch_size,
        gpus=args.gpus,
    )
    scores = predictions.scores
    if len(scores) != len(pairs):
        raise RuntimeError(f"COMET returned {len(scores)} scores for {len(pairs)} input segments")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(("source_sentence", "target_sentence", "cometkiwi_score"))
        writer.writerows((source, target, format(float(score), ".10g"))
                         for (source, target), score in zip(pairs, scores, strict=True))
    temporary_path.replace(output_path)
    print(f"Wrote {len(scores)} segment scores to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
