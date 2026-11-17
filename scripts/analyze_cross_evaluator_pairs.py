#!/usr/bin/env python3
"""Summarize directly reviewed same-error links without transitive clustering."""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets/nmt"
OUT = ROOT / "outputs"

HUMAN_FILES = {
    "Gemma": DATA / "gemma-4-31b-qat/accepted_human_gold_pairs.tsv",
    "Sol": DATA / "sol/accepted_human_gold_pairs_sol.tsv",
    "Luna": DATA / "luna/accepted_human_gold_pairs.tsv",
}
ACCEPTED_FILES = {
    "Gemma": DATA / "gemma-4-31b-qat/final_accepted.tsv",
    "Sol": DATA / "sol/final_accepted.tsv",
    "Luna": DATA / "luna/final_accepted.tsv",
}
LUNA_ACCEPTED = DATA / "luna/final_accepted.tsv"
HUMAN_CORPUS = DATA / "human_gold/anotovane_chyby_NMT.json"
MODEL_FILES = {
    ("Gemma", "Sol"): DATA / "FINAL gemma_sol_error_pairs.xlsx",
    ("Gemma", "Luna"): DATA / "FINAL gemma_luna_error_pairs.xlsx",
    ("Sol", "Luna"): DATA / "FINAL sol_luna_error_pairs.xlsx",
}


def normalize(value):
    text = unicodedata.normalize("NFKC", str(value or "")).replace("\u00a0", " ")
    return " ".join(text.split())


def source_key(value):
    return normalize(value).replace('"', "").replace("“", "").replace("”", "")


def model_finding_lookup_key(model, source, translation, source_span, target_span, comments):
    return (model, source_key(source), source_key(translation), normalize(source_span),
            normalize(target_span), normalize(comments))


def model_key(model, row):
    return model_finding_lookup_key(
        model.lower(), row["source_sentence"], row["translation"],
        row["model_source_span"], row["model_target_span"], row["model_error_comments"],
    )


def human_key(row):
    return source_key(row["source_sentence"]) + "\x1f" + normalize(row["human_error_comments"])


def tex_table(name, header, rows, alignment):
    lines = [r"\begin{tabular}{" + alignment + "}", r"\toprule",
             " & ".join(header) + " " + chr(92) * 2, r"\midrule"]
    lines.extend(" & ".join(map(str, row)) + r" \\" for row in rows)
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (OUT / f"{name}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_tsv(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def read_xlsx_sheet(path):
    """Read the first worksheet from these data-only XLSX review workbooks."""
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as workbook:
        shared = []
        if "xl/sharedStrings.xml" in workbook.namelist():
            root = ET.fromstring(workbook.read("xl/sharedStrings.xml"))
            shared = ["".join(item.itertext()) for item in root.findall("m:si", ns)]
        root = ET.fromstring(workbook.read("xl/worksheets/sheet1.xml"))

    def column_index(reference):
        letters = re.match(r"[A-Z]+", reference).group()
        value = 0
        for letter in letters:
            value = value * 26 + ord(letter) - ord("A") + 1
        return value - 1

    rows = []
    for row in root.findall(".//m:sheetData/m:row", ns):
        cells = {}
        for cell in row.findall("m:c", ns):
            kind = cell.get("t")
            if kind == "inlineStr":
                inline = cell.find("m:is", ns)
                value = "".join(inline.itertext()) if inline is not None else ""
            else:
                item = cell.find("m:v", ns)
                value = item.text if item is not None else ""
                if kind == "s" and value:
                    value = shared[int(value)]
            cells[column_index(cell.get("r", "A1"))] = value
        if cells:
            rows.append([cells.get(i, "") for i in range(max(cells) + 1)])
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    segment_88 = json.loads(HUMAN_CORPUS.read_text(encoding="utf-8"))[87]
    luna_88_keys = set()
    with LUNA_ACCEPTED.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            if (normalize(row["source_sentence"]) == normalize(segment_88["source_sentence"])
                    and normalize(row["target_sentence"]) == normalize(segment_88["translation"])):
                luna_88_keys.add(model_key("Luna", {
                    "source_sentence": row["source_sentence"],
                    "translation": row["target_sentence"],
                    "model_source_span": row["source_span"],
                    "model_target_span": row["target_span"],
                    "model_error_comments": row["justification"],
                }))
    if len(luna_88_keys) != 4:
        raise ValueError(f"Expected four recovered Luna findings in segment 88, found {len(luna_88_keys)}")
    human_links = {}
    human_summary = []
    for display, path in HUMAN_FILES.items():
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream, delimiter="\t"))
        with ACCEPTED_FILES[display].open(encoding="utf-8-sig", newline="") as stream:
            accepted_rows = list(csv.DictReader(
                stream, delimiter="\t"
            ))
        links = set()
        findings = set()
        negative_count = 0
        blank_count = 0
        positive_count = 0
        seen_pairs = set()
        for row in rows:
            mk = model_key(display, row)
            hk = human_key(row)
            findings.add(mk)
            decision = normalize(row["same_error_human_decision"])
            pair = (mk, hk)
            if pair in seen_pairs:
                raise ValueError(f"Duplicate human--model candidate pair in {path}: {pair!r}")
            seen_pairs.add(pair)
            if decision == "1":
                positive_count += 1
                links.add((mk, hk))
            elif decision == "0":
                negative_count += 1
            elif decision == "":
                blank_count += 1
            else:
                raise ValueError(f"Unexpected human-link decision {decision!r} in {path}")
        human_links[display] = links
        matched_findings = {m for m, h in links}
        resolved_links = {(m, h) for m, h in links if h.split("\x1f", 1)[1]}
        resolved_models = {m for m, h in resolved_links}
        linked_humans = {h for _, h in resolved_links}
        human_summary.append({
            "model": display, "accepted_findings": len(accepted_rows),
            "findings_in_pair_file": len(findings),
            "accepted_findings_without_pair_rows": len(accepted_rows) - len(findings),
            "candidate_pairs_assessed": len(rows), "positive_pairs": len(links),
            "positive_decision_rows": positive_count,
            "positive_pair_share": positive_count / len(rows) if rows else None,
            "explicit_negative_pairs": negative_count, "blank_decisions": blank_count,
            "findings_with_positive_link_including_blank_endpoint": len(matched_findings),
            "findings_with_identifiable_positive_link": len(resolved_models),
            "unresolved_positive_findings": len(matched_findings - resolved_models),
            "distinct_human_comment_units_linked": len(linked_humans),
        })

    model_links = {}
    model_summary = []
    luna_keys_by_pair = defaultdict(set)
    for (a, b), path in MODEL_FILES.items():
        sheet_rows = read_xlsx_sheet(path)
        headers = sheet_rows[0]
        rows = []
        candidate_count = 0
        negative_count = 0
        blank_count = 0
        positive_count = 0
        for values in sheet_rows[1:]:
            if not any(value is not None and str(value).strip() for value in values):
                continue
            candidate_count += 1
            row = {name: values[index] if index < len(values) else ""
                   for index, name in enumerate(headers)}
            def finding(model):
                return model_key(model, {
                    "source_sentence": row["source_sentence"],
                    "translation": row["translation"],
                    "model_source_span": row[f"{model.lower()}_source_span"],
                    "model_target_span": row[f"{model.lower()}_target_span"],
                    "model_error_comments": row[f"{model.lower()}_error_comments"],
                })
            left_finding, right_finding = finding(a), finding(b)
            if a == "Luna":
                luna_keys_by_pair[(a, b)].add(left_finding)
            if b == "Luna":
                luna_keys_by_pair[(a, b)].add(right_finding)
            decision = normalize(row["same_error_decision"])
            if decision == "1":
                positive_count += 1
                rows.append((left_finding, right_finding))
            elif decision in ("", "none"):
                blank_count += 1
            elif decision == "0":
                negative_count += 1
            else:
                raise ValueError(f"Unexpected decision {decision!r} in {path}")
        links = set(rows)
        model_links[(a, b)] = links
        model_summary.append({
            "model_pair": f"{a}--{b}", "positive_direct_pairs": len(links),
            "findings_with_positive_link_first": len({x for x, _ in links}),
            "findings_with_positive_link_second": len({y for _, y in links}),
            "candidate_sheet_rows": candidate_count,
            "positive_pair_share": positive_count / candidate_count if candidate_count else None,
            "positive_decision_rows": positive_count,
            "duplicate_positive_pair_rows": positive_count - len(links),
            "explicit_negative_decisions": negative_count,
            "blank_decisions": blank_count,
        })

    gs = model_links[("Gemma", "Sol")]
    gl = model_links[("Gemma", "Luna")]
    sl = model_links[("Sol", "Luna")]
    triangles = {(g, s, l) for g, s in gs for gg, l in gl
                 if gg == g and (s, l) in sl}
    for pair in (("Gemma", "Luna"), ("Sol", "Luna")):
        if not luna_88_keys.issubset(luna_keys_by_pair[pair]):
            raise ValueError(f"Not all four recovered Luna segment-88 findings appear in {pair} workbook")
    stats = {
        "human_model": human_summary,
        "model_model": model_summary,
        "directly_confirmed_model_triplets": len(triangles),
        "triplet_findings_by_model": {
            "gemma": len({g for g, _, _ in triangles}),
            "sol": len({s for _, s, _ in triangles}),
            "luna": len({l for _, _, l in triangles}),
        },
        "luna_segment_88_included": len(luna_88_keys),
        "luna_segment_88_included_in_model_pairs": {
            "Gemma--Luna": len(luna_88_keys & luna_keys_by_pair[("Gemma", "Luna")]),
            "Sol--Luna": len(luna_88_keys & luna_keys_by_pair[("Sol", "Luna")]),
        },
        "interpretation": "Counts are direct reviewed links; triplets require a positive direct decision for all three model pairs. Counts are not disjoint unique-error totals.",
        "source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [*HUMAN_FILES.values(), *ACCEPTED_FILES.values(), LUNA_ACCEPTED,
                         HUMAN_CORPUS, *MODEL_FILES.values()]
        },
    }
    (OUT / "cross_evaluator_direct_links.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_tsv(OUT / "cross_evaluator_human_model_links.tsv", human_summary)
    write_tsv(OUT / "cross_evaluator_model_model_links.tsv", model_summary)
    tex_table("cross_evaluator_human_model_links",
              ["Model", "Accepted", "Pair-file findings", "Same / rows (\\%)", "Findings linked", "Human units linked"],
              [[r["model"], r["accepted_findings"], r["findings_in_pair_file"],
                f'{r["positive_decision_rows"]}/{r["candidate_pairs_assessed"]} ({100*r["positive_pair_share"]:.1f}\\%)',
                r["findings_with_identifiable_positive_link"], r["distinct_human_comment_units_linked"]]
               for r in human_summary], "lrrrrr")
    tex_table("cross_evaluator_model_model_links",
              ["Model pair", "Same / rows (\\%)", "Findings linked (first)", "Findings linked (second)"],
              [[r["model_pair"],
                f'{r["positive_decision_rows"]}/{r["candidate_sheet_rows"]} ({100*r["positive_pair_share"]:.1f}\\%)',
                r["findings_with_positive_link_first"],
                r["findings_with_positive_link_second"]]
               for r in model_summary], "lrrr")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
