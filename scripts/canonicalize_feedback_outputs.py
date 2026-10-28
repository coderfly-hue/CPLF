#!/usr/bin/env python3
"""Create manuscript-compliant per-case feedback evidence.

The manuscript specifies that the independent evaluator returns an integer score
from 1 to 5 for diagnostic consistency, actionability, personalization and
answer leakage, followed by linear normalization to [0,1].  This exporter keeps
both the raw integer and normalized score and recomputes FQS from the per-case
components.  It intentionally refuses arbitrary continuous rubric scores.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

RAW_TO_NORM = {1:0.0, 2:0.25, 3:0.5, 4:0.75, 5:1.0}
NORM_TO_RAW = {v:k for k,v in RAW_TO_NORM.items()}
DIMS = {
    "diagnostic": ("diagnostic_raw", "diagnostic_consistency"),
    "actionability": ("actionability_raw", "actionability"),
    "personalization": ("personalization_raw", "personalization"),
    "leakage": ("leakage_raw", "answer_leakage"),
}

def _raw_candidates(df: pd.DataFrame, dim: str, raw_col: str) -> list[str]:
    candidates=[raw_col]
    if dim == "diagnostic": candidates += ["diagnostic_consistency_raw"]
    if dim == "leakage": candidates += ["answer_leakage_raw"]
    return [c for c in candidates if c in df.columns]

def derive_raw_and_norm(df: pd.DataFrame, dim: str, raw_col: str, norm_col: str) -> tuple[pd.Series,pd.Series]:
    raw_candidates=_raw_candidates(df,dim,raw_col)
    if not raw_candidates:
        raise ValueError(
            f"Missing original evaluator integer field {raw_col}; do not infer raw "
            "ratings from normalized legacy scores; do not round legacy scores."
        )
    raw=pd.to_numeric(df[raw_candidates[0]],errors="raise")
    if not np.all(np.isclose(raw, np.round(raw))):
        raise ValueError(f"{raw_candidates[0]} must contain integer evaluator scores 1..5")
    raw=raw.astype(int)
    bad=~raw.isin(RAW_TO_NORM)
    if bad.any():
        raise ValueError(f"{raw_candidates[0]} contains scores outside 1..5")
    norm=raw.map(RAW_TO_NORM).astype(float)
    if norm_col in df.columns:
        supplied=pd.to_numeric(df[norm_col],errors="coerce")
        mismatch=~np.isclose(supplied,norm,atol=1e-9,equal_nan=False)
        if mismatch.any():
            raise ValueError(f"{norm_col} disagrees with normalized raw evaluator scores")
    return raw,norm

def fqs(r,d,a,p,l):
    return np.maximum(0.0, 0.25*r + 0.25*d + 0.25*a + 0.25*p - 0.40*l)

def canonicalize(df: pd.DataFrame) -> pd.DataFrame:
    required=["dataset","case_id","method","seed","relevance"]
    missing=[c for c in required if c not in df.columns]
    if missing: raise ValueError(f"Missing required columns: {missing}")
    out=df.copy()
    out["relevance"]=pd.to_numeric(out["relevance"],errors="raise").clip(0,1)
    for dim,(raw_col,norm_col) in DIMS.items():
        raw,norm=derive_raw_and_norm(out,dim,raw_col,norm_col)
        out[raw_col]=raw
        out[norm_col]=norm
    for field in ("feedback_final", "evaluator_raw_response"):
        if field not in out.columns or out[field].isna().any() or not out[field].astype(str).str.strip().all():
            raise ValueError(f"Canonical feedback requires original nonempty {field}")
    out["fqs"]=fqs(out.relevance,out.diagnostic_consistency,out.actionability,out.personalization,out.answer_leakage)
    # Stable column order; retain additional provenance columns afterwards.
    first=[c for c in [
        "dataset","case_id","learner_id","method","seed",
        "feedback_before_verification","feedback_final","feedback_text","evaluator_raw_response",
        "relevance","diagnostic_raw","actionability_raw","personalization_raw","leakage_raw",
        "diagnostic_consistency","actionability","personalization","answer_leakage",
        "before_fqs","fqs","revision_applied","accepted_on_first_pass","accepted_after_verification",
        "generator_model","generator_revision","evaluator_model","evaluator_resolved_revision"] if c in out.columns]
    rest=[c for c in out.columns if c not in first]
    return out[first+rest]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",required=True)
    ap.add_argument("--output",required=True)
    ap.add_argument("--expect-rows",type=int)
    args=ap.parse_args()
    df=pd.read_csv(args.input)
    out=canonicalize(df)
    if args.expect_rows is not None and len(out)!=args.expect_rows:
        raise SystemExit(f"Expected {args.expect_rows} rows, found {len(out)}")
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    out.to_csv(args.output,index=False)
    print(f"Wrote {len(out):,} canonical feedback rows to {args.output}")

if __name__=="__main__": main()
