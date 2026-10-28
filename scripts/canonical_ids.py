#!/usr/bin/env python3
"""Create a stable learner/session ID namespace shared by splits and predictions."""
from __future__ import annotations
import argparse, hashlib
from pathlib import Path
import pandas as pd

def canonical_id(dataset,source_id,kind="learner"):
    token=f"{dataset}\0{kind}\0{source_id}".encode("utf-8")
    return f"CID_{hashlib.sha256(token).hexdigest()[:20].upper()}"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",required=True)
    ap.add_argument("--output",required=True)
    ap.add_argument("--dataset-column",default="dataset")
    ap.add_argument("--source-id-column",required=True)
    ap.add_argument("--output-id-column",default="learner_id")
    ap.add_argument("--kind",default="learner",choices=["learner","session"])
    args=ap.parse_args()
    df=pd.read_csv(args.input)
    for c in [args.dataset_column,args.source_id_column]:
        if c not in df.columns: raise SystemExit(f"Missing {c}")
    out=df.copy()
    out[args.output_id_column]=[
        canonical_id(ds,sid,args.kind) for ds,sid in zip(out[args.dataset_column].astype(str),out[args.source_id_column].astype(str))
    ]
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    out.to_csv(args.output,index=False)
    print(f"Wrote stable canonical IDs for {len(out):,} rows")
if __name__=="__main__": main()
