#!/usr/bin/env python3
"""Reject proxy ablation exports and build canonical prediction-side evidence."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",required=True)
    ap.add_argument("--output",required=True)
    args=ap.parse_args()
    df=pd.read_csv(args.input)
    proxy=[c for c in df.columns if "proxy" in c.lower()]
    if proxy:
        raise SystemExit(
            "Proxy columns are not canonical experimental evidence: " + ", ".join(proxy) +
            ". Export genuine held-out y_true/y_prob rows from every ablation run."
        )
    req=["dataset","case_id","variant","seed","y_true","y_prob"]
    miss=[c for c in req if c not in df.columns]
    if miss: raise SystemExit(f"Missing canonical ablation columns: {miss}")
    if not set(df.y_true.dropna().astype(int).unique()).issubset({0,1}): raise SystemExit("y_true must be binary")
    if ((df.y_prob<0)|(df.y_prob>1)).any(): raise SystemExit("y_prob must be in [0,1]")
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(args.output,index=False)
    print(f"Wrote {len(df):,} canonical ablation prediction rows")
if __name__=="__main__": main()
