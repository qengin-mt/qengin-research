#!/usr/bin/env python3
"""Reproduce manuscript analyses from original artifacts; never edit source data."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import importlib.metadata
import itertools
import json
import math
from pathlib import Path
import platform
import sys
import unicodedata

import numpy as np
from scipy.stats import rankdata
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets/nmt"
OUT = ROOT / "outputs"
MODELS = {"gemma": "gemma-4-31b-qat", "sol": "sol", "luna": "luna"}
FIELDS = ("category", "subtype", "severity", "source_span", "target_span", "justification")
WEIGHTS = {"neutral": 0, "minor": .25, "major": .5, "critical": 1}
INPUTS = set()


def read_json(path):
    INPUTS.add(path)
    return json.loads(path.read_text(encoding="utf-8"))


def read_table(path):
    INPUTS.add(path)
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def norm(value):
    return " ".join(unicodedata.normalize("NFKC", str(value or "")).split())


def join_norm(value):
    # Historical spreadsheet exports escaped quotes inconsistently.
    return norm(value).replace('"', "").replace("“", "").replace("”", "")


def judgment(value):
    if value in ("1", "accept"):
        return 1
    if value in ("0", "reject"):
        return 0
    if value == "":
        return None
    raise ValueError(f"Unknown decision {value!r}")


def key(segment, error):
    return (segment, *(join_norm(error[f]) for f in FIELDS))


def wilson(k, n):
    if not n:
        return [None, None]
    z = 1.959963984540054
    p = k / n
    centre = (p + z*z/(2*n)) / (1+z*z/n)
    radius = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n)) / (1+z*z/n)
    return [centre-radius, centre+radius]


def validation(rows):
    n = len(rows)
    accepted = sum(r["final"] for r in rows)
    paired = [r for r in rows if r["a1"] is not None and r["a2"] is not None]
    agreement = sum(r["a1"] == r["a2"] for r in paired)
    a = sum(r["a1"] for r in paired)
    b = sum(r["a2"] for r in paired)
    pn = len(paired)
    expected = (a*b+(pn-a)*(pn-b))/(pn*pn) if pn else 1
    kappa = (agreement/pn-expected)/(1-expected) if expected != 1 else None
    disagreements = [r for r in paired if r["a1"] != r["a2"]]
    aa = sum(r["a1"] == 1 and r["a2"] == 1 for r in paired)
    ar = sum(r["a1"] == 1 and r["a2"] == 0 for r in paired)
    ra = sum(r["a1"] == 0 and r["a2"] == 1 for r in paired)
    rr = sum(r["a1"] == 0 and r["a2"] == 0 for r in paired)
    positive_agreement = 2*aa/(2*aa+ar+ra) if 2*aa+ar+ra else None
    negative_agreement = 2*rr/(2*rr+ar+ra) if 2*rr+ar+ra else None
    assert all(r["a3"] is not None and r["final"] == r["a3"] for r in disagreements)
    assert all(r["final"] == r["a1"] for r in paired if r["a1"] == r["a2"])
    return dict(reviewed=n, accepted=accepted, rejected=n-accepted,
                acceptance=accepted/n if n else None, wilson95=wilson(accepted,n),
                a1a2_n=pn, agree=agreement, agreement_rate=agreement/pn if pn else None,
                confusion=dict(a1_accept_a2_accept=aa, a1_accept_a2_reject=ar,
                               a1_reject_a2_accept=ra, a1_reject_a2_reject=rr),
                positive_agreement=positive_agreement, negative_agreement=negative_agreement,
                kappa=kappa, adjudicated=len(disagreements),
                extra_a3_agreement_rows=sum(r["a3"] is not None and r["a1"]==r["a2"] for r in paired))


def corr(x, y):
    x, y = np.asarray(x), np.asarray(y)
    x = x-x.mean(axis=-1, keepdims=True)
    y = y-y.mean(axis=-1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.sum(x*y, axis=-1)/np.sqrt(np.sum(x*x, axis=-1)*np.sum(y*y, axis=-1))


def interval(values):
    values = np.asarray(values)
    return [float(x) for x in np.quantile(values[np.isfinite(values)], [.025,.975])]


def distribution(x):
    return dict(zip(("min","q25","median","q75","max"),map(float,np.quantile(x,[0,.25,.5,.75,1]))))


def tex(value):
    return "".join({"\\":r"\textbackslash{}", "&":r"\&", "%":r"\%", "_":r"\_",
                    "#":r"\#", "$":r"\$", "{":r"\{", "}":r"\}"}.get(c,c) for c in str(value))


def table(name, header, rows, alignment=None):
    alignment = alignment or ("l"+"r"*(len(header)-1))
    lines=[r"\begin{tabular}{"+alignment+"}",r"\toprule",
           " & ".join(header)+r" \\",r"\midrule"]
    lines += [" & ".join(map(str,r))+r" \\" for r in rows]
    lines += [r"\bottomrule",r"\end{tabular}"]
    (OUT/(name+".tex")).write_text("\n".join(lines)+"\n",encoding="utf-8")


def write_tsv(name, rows):
    if not rows: return
    with (OUT/name).open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter="\t")
        w.writeheader(); w.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap",type=int,default=10000)
    parser.add_argument("--seed",type=int,default=20261012)
    parser.add_argument("--cometkiwi-scores",type=Path,
                        default=DATA/"comet_wmt22_cometkiwi_da_vs_qe_segments.tsv",
                        help="Segment-level COMETKiwi TSV with source_sentence, target_sentence, cometkiwi_score")
    args=parser.parse_args()
    if args.bootstrap<100: raise ValueError("Use at least 100 bootstrap replicates")
    OUT.mkdir(parents=True,exist_ok=True)
    gold=read_json(DATA/"human_gold/anotovane_chyby_NMT.json")
    source={join_norm(r["source_sentence"]):s for s,r in enumerate(gold,1)}
    assert len(source)==len(gold), "Ambiguous normalized source"
    human=defaultdict(list)
    for s,r in enumerate(gold,1):
        for j,e in enumerate(r["errors"],1):
            c=e.get("comments",[])
            c=" | ".join(map(str,c)) if isinstance(c,list) else str(c)
            if norm(c): human[(s,norm(c))].append((j,e.get("id")))
    # Quote normalization must not coalesce distinct canonical comment units.
    strict={(s,norm(" | ".join(e["comments"]) if isinstance(e["comments"],list) else e["comments"]))
            for s,r in enumerate(gold,1) for e in r["errors"] if e.get("comments")}
    assert len(strict)==len(human)
    errors_by_segment={}; completed_errors_by_segment={}; labels={}; links={}; audits={}; unresolved_by_model={}; all_reviews=[]; pair_audit=[]
    prediction_keys={}; prediction_segments={}
    for m,d in MODELS.items():
        raw_records=read_json(DATA/d/"model_results.json")
        assert len(raw_records)==len(gold)
        prediction_keys[m]=set()
        prediction_segments[m]=set()
        errors_by_segment[m]={s:[] for s in range(1,len(gold)+1)}
        completed_errors_by_segment[m]={s:[] for s in range(1,len(gold)+1)}
        raw_by_segment={}
        for s,raw in enumerate(raw_records,1):
            assert join_norm(raw["source_sentence"])==join_norm(gold[s-1]["source_sentence"])
            assert join_norm(raw["translation"])==join_norm(gold[s-1]["translation"])
            response=raw.get("qe_engine")
            if response is None:
                continue
            prediction_segments[m].add(s)
            raw_by_segment[s]=[]
            for raw_error in response.get("errors",[]):
                error={f:raw_error[f] for f in FIELDS}
                k=key(s,error)
                assert k not in prediction_keys[m], f"Duplicate raw prediction: {m} segment {s}"
                prediction_keys[m].add(k)
                raw_by_segment[s].append(error)
                errors_by_segment[m][s].append(error)
        record_keys={}
        labels[m]={}
        for row in read_table(DATA/d/("final_labels.tsv")):
            s=source[join_norm(row["source_sentence"])]
            def field(f): return row[("model_error_comments" if f=="justification" else "model_"+f) if m=="gemma" else f]
            e={f:field(f) for f in FIELDS}
            k=key(s,e)
            assert k not in labels[m]
            a=[judgment(row[f"annotator_{i}_"+("decision" if m=="gemma" else "judgment")]) for i in (1,2,3)]
            final=judgment(row["final_label" if m=="gemma" else "final_judgment"])
            assert final is not None
            assert join_norm(row["translation" if m=="gemma" else "target_sentence"])==join_norm(gold[s-1]["translation"])
            completed_errors_by_segment[m][s].append(e)
            record_keys[k]=(s,len(completed_errors_by_segment[m][s]))
            source_tag=row["final_label_source"]
            if m == "luna" and source_tag == "gold_annotation":
                source_tag="manual_luna_gpt_recovery"
            entry=dict(model=m,segment=s,dataset_error_index=record_keys[k][1],
                       a1=a[0],a2=a[1],a3=a[2],final=final,
                       source=source_tag,**e)
            labels[m][k]=entry
            all_reviews.append(entry)

        supplemental_keys={k for k,r in labels[m].items() if r["source"]=="manual_luna_gpt_recovery"}
        assert set(labels[m])-supplemental_keys==prediction_keys[m], (m,"raw predictions do not match final research rows")
        assert supplemental_keys<=set(labels[m])
        errors_by_segment[m]=raw_by_segment
        # The four Luna segment-88 rows are manually recovered Luna GPT
        # findings; they are model findings even though the failed API call
        # left no response in the saved JSON.
        model_labels=dict(labels[m])
        main_rows=[r for r in model_labels.values() if r["a1"] is not None and r["a2"] is not None]
        final_labeled=list(model_labels.values())
        accepted_rows=read_table(DATA/d/("final_accepted.tsv"))
        accepted_keys=[]
        for row in accepted_rows:
            seg=source[join_norm(row["source_sentence"])]
            assert join_norm(row["translation" if m=="gemma" else "target_sentence"])==join_norm(gold[seg-1]["translation"])
            assert judgment(row["final_label" if m=="gemma" else "final_judgment"])==1
            err={f:row[("model_error_comments" if f=="justification" else "model_"+f) if m=="gemma" else f] for f in FIELDS}
            k=key(seg,err)
            assert k in labels[m] and labels[m][k]["final"]==1
            accepted_keys.append(k)
        assert len(accepted_keys)==len(set(accepted_keys))
        assert set(accepted_keys)=={k for k,r in labels[m].items() if r["final"]}
        accepted_model_keys=set(accepted_keys)

        links[m]=defaultdict(set)
        pairfile=DATA/d/("accepted_human_gold_pairs_sol.tsv" if m=="sol" else "accepted_human_gold_pairs.tsv")
        pairrows=read_table(pairfile)
        seen=set(); candidates=defaultdict(set); unresolved=set()
        for index,row in enumerate(pairrows,2):
            s=source[join_norm(row["source_sentence"])]
            assert join_norm(row["translation"])==join_norm(gold[s-1]["translation"])
            e={f:row["model_error_comments" if f=="justification" else "model_"+f] for f in FIELDS}
            k=key(s,e)
            assert k in labels[m] and labels[m][k]["final"]==1
            h=(s,norm(row["human_error_comments"]))
            dec=judgment(row["same_error_human_decision"])
            assert dec is not None
            # Empty placeholder for a segment without a non-empty gold comment.
            assert h in human or not h[1], (m,index,h,dec)
            assert (k,h) not in seen
            seen.add((k,h)); candidates[k].add(h)
            if dec and h in human: links[m][k].add(h)
            if dec and not h[1]: unresolved.add(k)
            pair_audit.append(dict(model=m,pair_file_row=index,segment=s,
                dataset_error_index=labels[m][k]["dataset_error_index"],decision=dec,
                human_comment=h[1],original_annotation_indices=",".join(str(j) for j,_ in human.get(h,[]))))
        accepted=set(accepted_keys)
        pending_pairs=accepted-set(candidates)
        unresolved_by_model[m]=unresolved|pending_pairs
        assert set(candidates)<=accepted
        for k in candidates:
            expected={h for h in human if h[0]==k[0]} or {(k[0],"")}
            # A later, researcher-confirmed positive link can be recorded
            # without fabricating decisions for the other same-segment human
            # descriptions. A positive link is sufficient to establish that
            # the accepted model finding has a human correspondence.
            assert candidates[k]==expected or bool(links[m][k]), "Candidate matching is incomplete and has no confirmed positive link"
        def matching(keys):
            matched=sum(bool(links[m][k]) for k in keys)
            covered=set().union(*(links[m][k] for k in keys))
            represented=sum(len(human[h]) for h in covered)
            pending=sum((k in unresolved or k in pending_pairs) and not links[m][k] for k in keys)
            return dict(accepted=len(keys),matched=matched,unmatched=len(keys)-matched-pending,unresolved=pending,
                        correspondence=matched/len(keys),wilson95=wilson(matched,len(keys)),
                        covered_comment_units=len(covered),eligible_comment_units=len(human),
                        coverage=len(covered)/len(human),coverage_wilson95=wilson(len(covered),len(human)),
                        original_records_in_linked_comment_groups=represented)
        audits[m]=dict(corpus_segments=len(gold),model_output_segments=len(prediction_segments[m]),
            model_prediction_rows=len(prediction_keys[m]),manually_recovered_rows=len(supplemental_keys),
            research_table_rows=len(record_keys),final_rows=len(model_labels),
            primary=validation(final_labeled),paired_review=validation(main_rows),
            label_sources=dict(Counter(r["source"] for r in labels[m].values())),
            matching_primary=matching(accepted_model_keys),
            pair_rows=len(pairrows),
            positive_pair_rows=sum(bool(judgment(r["same_error_human_decision"])) for r in pairrows),
            pending_pair_reviews=[dict(segment=labels[m][k]["segment"],error_index=labels[m][k]["dataset_error_index"]) for k in pending_pairs if k in prediction_keys[m]])
    baseline={}
    original=read_table(args.cometkiwi_scores)
    for r in original:
        seg=source[join_norm(r["source_sentence"])]
        assert join_norm(r["target_sentence"])==join_norm(gold[seg-1]["translation"])
        assert seg not in baseline
        baseline[seg]=float(r["cometkiwi_score"])
    assert set(baseline)==set(range(1,len(gold)+1))
    common=sorted(set(baseline).intersection(*(prediction_segments[m] for m in MODELS)))
    volumes={seg:len(gold[seg-1]["translation"].split()) for seg in common}
    assert min(volumes.values())>0
    scores={m:{seg:1/(1+sum(WEIGHTS[e["severity"]] for e in errors_by_segment[m].get(seg,[]))/volumes[seg])
               for seg in common} for m in MODELS}
    rng=np.random.default_rng(args.seed)
    indices=rng.integers(0,len(common),(args.bootstrap,len(common)))
    y=np.array([baseline[s] for s in common])
    yb=y[indices]; yr=rankdata(yb,axis=1)
    score_results={}; boot={}
    for m in MODELS:
        x=np.array([scores[m][s] for s in common]); xb=x[indices]
        pb=corr(xb,yb); rb=corr(rankdata(xb,axis=1),yr); boot[m]=pb

        score_results[m]=dict(n=len(common),pearson=float(corr(x,y)),pearson_ci95=interval(pb),
            spearman=float(corr(rankdata(x),rankdata(y))),spearman_ci95=interval(rb),
            invalid_bootstrap=int(np.sum(~np.isfinite(pb))),distribution=distribution(x),
            error_free_segments=sum(not errors_by_segment[m][s] for s in common),
            q_length_pearson=float(corr(x,[volumes[s] for s in common])))
    comparisons={a+"-"+b:dict(delta=score_results[a]["pearson"]-score_results[b]["pearson"],
                            ci95=interval(boot[a]-boot[b])) for a,b in itertools.combinations(MODELS,2)}
    sensitivity=[]
    profiles={"active":WEIGHTS,"equal":{"neutral":0,"minor":1,"major":1,"critical":1},
              "1_2_4":{"neutral":0,"minor":1,"major":2,"critical":4},
              "2_4_8":{"neutral":0,"minor":2,"major":4,"critical":8}}
    for m in MODELS:
        for name,w in profiles.items():
            e=np.array([sum(w[error["severity"]] for error in errors_by_segment[m][s])/volumes[s] for s in common])
            for scale in ((.1,.5,1,4) if name=="active" else ((1,4) if name=="1_2_4" else (1,))):
                x=scale/(e+scale)
                sensitivity.append(dict(model=m,profile=name,S=scale,pearson=float(corr(x,y)),
                                        spearman=float(corr(rankdata(x),rankdata(y)))))
    # Mathematical checks catch a reversed score or accidental rounding of ranks.
    for m in MODELS:
        rows=[r for r in sensitivity if r["model"]==m and r["profile"]=="active"]
        assert max(r["spearman"] for r in rows)-min(r["spearman"] for r in rows)<1e-12
        a=next(r for r in sensitivity if r["model"]==m and r["profile"]=="1_2_4" and r["S"]==4)
        assert abs(a["pearson"]-score_results[m]["pearson"])<1e-12
    by_category=[]; by_severity=[]
    for m in MODELS:
        primary=[r for k,r in labels[m].items() if k in prediction_keys[m]]
        for field,target in (("category",by_category),("severity",by_severity)):
            for value in sorted({r[field] for r in primary}):
                sub=[r for r in primary if r[field]==value]
                target.append(dict(model=m,group=value,reviewed=len(sub),accepted=sum(r["final"] for r in sub),
                                   acceptance=sum(r["final"] for r in sub)/len(sub)))
    cases=[]
    for m in MODELS:
        for case in ("accepted_matched","accepted_unmatched","rejected"):
            for k,r in labels[m].items():
                if k not in prediction_keys[m]: continue
                if r["dataset_error_index"] is None or r["a1"] is None or r["a2"] is None: continue
                status="rejected" if not r["final"] else ("accepted_matched" if links[m][k] else ("unresolved" if k in unresolved_by_model[m] else "accepted_unmatched"))
                if status==case:
                    cases.append(dict(case=case,**r,source_sentence=gold[r["segment"]-1]["source_sentence"],
                                      translation=gold[r["segment"]-1]["translation"],
                                      human_comments=" | ".join(h[1] for h in sorted(links[m][k]))))
                    break
    segment_rows=[]
    for s in common:
        row=dict(segment=s,source_sentence=gold[s-1]["source_sentence"],translation=gold[s-1]["translation"],
                 target_words=volumes[s],cometkiwi=baseline[s])
        for m in MODELS:
            p=sum(WEIGHTS[e["severity"]] for e in errors_by_segment[m][s])
            row.update({m+"_P":p,m+"_E":p/volumes[s],m+"_Q":scores[m][s],m+"_accepted_Q":1/(1+sum(WEIGHTS[r["severity"]] for k,r in labels[m].items() if k in prediction_keys[m] and r["segment"]==s and r["final"])/volumes[s])})
        segment_rows.append(row)
    extremes=[]
    for m in MODELS:
        # Compare ranks, as the numerical score scales are not calibrated together.
        x=np.array([scores[m][s] for s in common])
        gaps=np.abs(rankdata(x)-rankdata(y))
        for index in np.argsort(-gaps,kind="stable")[:3]:
            s=common[index]
            extremes.append(dict(model=m,segment=s,rank_gap=float(gaps[index]),
                                 Q=scores[m][s],cometkiwi=baseline[s],
                                 source_sentence=gold[s-1]["source_sentence"],translation=gold[s-1]["translation"]))
    completed_135_scores={}
    full_segments=sorted(baseline)
    full_y=np.array([baseline[s] for s in full_segments])
    full_indices=np.random.default_rng(args.seed).integers(0,len(full_segments),(args.bootstrap,len(full_segments)))
    for m in MODELS:
        full_x=np.array([1/(1+sum(WEIGHTS[e["severity"]] for e in completed_errors_by_segment[m][s])/len(gold[s-1]["translation"].split())) for s in full_segments])
        full_xb=full_x[full_indices]; full_yb=full_y[full_indices]
        completed_135_scores[m]=dict(n=len(full_segments),pearson=float(corr(full_x,full_y)),
            pearson_ci95=interval(corr(full_xb,full_yb)),
            spearman=float(corr(rankdata(full_x),rankdata(full_y))),
            spearman_ci95=interval(corr(rankdata(full_xb,axis=1),rankdata(full_yb,axis=1))))
    result=dict(bootstrap_replicates=args.bootstrap,seed=args.seed,
        dataset=dict(segments=len(gold),unique_sources=len(source),unique_pairs=len({(r["source_sentence"],r["translation"]) for r in gold}),
                     original_human_annotations=sum(len(r["errors"]) for r in gold),
                     eligible_comment_units=len(human),comment_multiplicities=dict(Counter(map(len,human.values()))),
                     corpus_target_words=distribution([len(r["translation"].split()) for r in gold]),
                     comparison_target_words=distribution(list(volumes.values())),baseline=distribution(list(baseline.values()))),
        common_segments=common,models=audits,scores=score_results,
        completed_table_135_scores=completed_135_scores,paired_pearson_differences=comparisons,
        sensitivity=sensitivity,by_category=by_category,by_severity=by_severity)
    (OUT/"analysis.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    write_tsv("review_audit.tsv",all_reviews)
    write_tsv("pair_audit.tsv",pair_audit)
    write_tsv("segment_scores.tsv",segment_rows)
    write_tsv("qualitative_cases.tsv",cases)
    write_tsv("rank_disagreements.tsv",extremes)
    write_tsv("sensitivity.tsv",sensitivity)
    write_tsv("by_category.tsv",by_category); write_tsv("by_severity.tsv",by_severity)
    fmtci=lambda x: f"[{x[0]:.3f}, {x[1]:.3f}]"
    pctci=lambda k,n,ci:f"{100*k/n:.1f} [{100*ci[0]:.1f}, {100*ci[1]:.1f}]"
    table("validation",["Model","Model findings","Accept","Reject",r"Accept \% [95\% CI]"],
          [[m.title(),a["final_rows"],a["primary"]["accepted"],a["primary"]["rejected"],
            pctci(a["primary"]["accepted"],a["primary"]["reviewed"],a["primary"]["wilson95"])] for m,a in audits.items()])
    table("agreement",["Model","A1/A2 pairs",r"Agree (\%)","Adjudicated",r"$\kappa$"],
          [[m.title(),a["primary"]["a1a2_n"],f'{a["primary"]["agree"]} ({100*a["primary"]["agreement_rate"]:.1f})',
            a["primary"]["adjudicated"],f'{a["primary"]["kappa"]:.3f}'] for m,a in audits.items()])
    table("agreement_confusion",
          ["Model", "A1=1/A2=1", "A1=1/A2=0", "A1=0/A2=1", "A1=0/A2=0",
           "Positive agreement (\\%)", "Negative agreement (\\%)"],
          [[m.title(), a["primary"]["confusion"]["a1_accept_a2_accept"],
            a["primary"]["confusion"]["a1_accept_a2_reject"],
            a["primary"]["confusion"]["a1_reject_a2_accept"],
            a["primary"]["confusion"]["a1_reject_a2_reject"],
            f'{100*a["primary"]["positive_agreement"]:.1f}',
            f'{100*a["primary"]["negative_agreement"]:.1f}'] for m,a in audits.items()],
          alignment="lrrrrrr")
    table("matching",["Model","Accepted","Matched","Unmatched","Unresolved","Covered/eligible",r"Coverage \%"],
          [[m.title(),a["matching_primary"]["accepted"],a["matching_primary"]["matched"],a["matching_primary"]["unmatched"],a["matching_primary"]["unresolved"],
            f'{a["matching_primary"]["covered_comment_units"]}/{len(human)}',f'{100*a["matching_primary"]["coverage"]:.1f}'] for m,a in audits.items()])
    table("matching_intervals",["Model",r"Correspondence \% [95\% CI]",r"Coverage \% [95\% CI]"],
          [[m.title(),pctci(a["matching_primary"]["matched"],a["matching_primary"]["accepted"],a["matching_primary"]["wilson95"]),
            pctci(a["matching_primary"]["covered_comment_units"],len(human),a["matching_primary"]["coverage_wilson95"])] for m,a in audits.items()])
    table("correlations",["Model","N",r"Pearson $r$ [95\% CI]",r"Spearman $\rho$ [95\% CI]"],
          [[m.title(),a["n"],f'{a["pearson"]:.3f} {fmtci(a["pearson_ci95"])}',f'{a["spearman"]:.3f} {fmtci(a["spearman_ci95"])}'] for m,a in score_results.items()])
    table("completed_table_135",["Population","N",r"Pearson $r$ [95\% CI]",r"Spearman $\rho$ [95\% CI]"],
          [["Luna recovered-output" if m=="luna" else m.title()+" saved-output",a["n"],
            f'{a["pearson"]:.3f} {fmtci(a["pearson_ci95"])}',f'{a["spearman"]:.3f} {fmtci(a["spearman_ci95"])}'] for m,a in completed_135_scores.items()],
          alignment="p{.38\\linewidth}rrr")
    table("paired_differences",["Contrast",r"$\Delta r$",r"95\% paired bootstrap CI"],
          [[k.title(),f'{v["delta"]:.3f}',fmtci(v["ci95"])] for k,v in comparisons.items()])
    profile_labels={"active":"Active","equal":"Equal","1_2_4":"1/2/4","2_4_8":"2/4/8"}
    sensitivity_rows=[]
    for p,s in dict.fromkeys((r["profile"],r["S"]) for r in sensitivity):
        label=profile_labels[p]
        if p=="active" and s==1:
            label="Active*"
        sensitivity_rows.append([label,s,*[
            f'{next(r["pearson"] for r in sensitivity if r["model"]==m and r["profile"]==p and r["S"]==s):.3f}'
            for m in MODELS]])
    table("sensitivity",["Severity profile (minor/major/critical)","S","Gemma $r$","Sol $r$","Luna $r$"],
          sensitivity_rows)
    table("by_severity",["Model","Severity","Reviewed","Accepted",r"Accept \%"],
          [[r["model"].title(),r["group"],r["reviewed"],r["accepted"],f'{100*r["acceptance"]:.1f}'] for r in by_severity],"llrrr")
    table("by_category",["Model","Category","Reviewed","Accepted",r"Accept \%"],
          [[r["model"].title(),tex(r["group"]),r["reviewed"],r["accepted"],f'{100*r["acceptance"]:.1f}'] for r in by_category],"llrrr")
    fig,axs=plt.subplots(1,3,figsize=(10.2,3.2),sharex=True,sharey=True,layout="constrained")
    for ax,m in zip(axs,MODELS):
        x=[scores[m][s] for s in common]
        ax.scatter(x,y,s=18,alpha=.6,color="#235789",edgecolors="none")
        ax.set(title=f'{m.title()} (N={len(common)})',xlabel="Quality Q from complete error-annotation set")
        ax.grid(alpha=.2)
    axs[0].set_ylabel("COMETKiwi score")
    fig.savefig(OUT/"score_scatter.pdf")
    plt.close(fig)
    INPUTS.add(Path(__file__))
    manifest=dict(python=platform.python_version(),packages={p:importlib.metadata.version(p) for p in ("numpy","scipy","matplotlib")},
                  inputs={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(INPUTS)},
                  note="Hashes of analysis inputs; no credentials or individual respondent metadata exported.")
    (OUT/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps({m:dict(validation=a["primary"],matching=a["matching_primary"],pending_pair_reviews=a["pending_pair_reviews"]) for m,a in audits.items()},indent=2,ensure_ascii=False))
    print(json.dumps(score_results,indent=2))


if __name__=="__main__":
    main()
