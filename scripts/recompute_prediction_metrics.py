#!/usr/bin/env python3
"""Recompute seed-level AUC/F1/NLL/Brier/ECE from per-case y_true/y_prob."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, log_loss, brier_score_loss

def expected_calibration_error(y_true,y_prob,n_bins=10):
    y=np.asarray(y_true,dtype=float); p=np.asarray(y_prob,dtype=float)
    bins=np.linspace(0.0,1.0,n_bins+1); total=0.0
    for i in range(n_bins):
        if i==n_bins-1: mask=(p>=bins[i]) & (p<=bins[i+1])
        else: mask=(p>=bins[i]) & (p<bins[i+1])
        if mask.any(): total += mask.mean()*abs(y[mask].mean()-p[mask].mean())
    return float(total)

def compute_group(df,n_bins):
    y=df.y_true.astype(int).to_numpy(); p=df.y_prob.astype(float).to_numpy()
    if not set(np.unique(y)).issubset({0,1}): raise ValueError("y_true must be binary")
    if ((p<0)|(p>1)).any(): raise ValueError("y_prob must be in [0,1]")
    return {
        "AUC":float(roc_auc_score(y,p)),
        "F1":float(f1_score(y,p>=0.5,zero_division=0)),
        "NLL":float(log_loss(y,np.clip(p,1e-12,1-1e-12),labels=[0,1])),
        "Brier":float(brier_score_loss(y,p)),
        "ECE":expected_calibration_error(y,p,n_bins),
        "n_cases":int(len(df)),
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",required=True)
    ap.add_argument("--seed-output",required=True)
    ap.add_argument("--summary-output",required=True)
    ap.add_argument("--ece-bins",type=int,default=10)
    ap.add_argument("--group-method-column",default="method",choices=["method","variant"])
    args=ap.parse_args()
    df=pd.read_csv(args.input)
    method=args.group_method_column
    req=["dataset",method,"seed","y_true","y_prob"]
    miss=[c for c in req if c not in df.columns]
    if miss: raise SystemExit(f"Missing required columns {miss}; proxy scores cannot reproduce AUC/ECE")
    rows=[]
    for keys,g in df.groupby(["dataset",method,"seed"],sort=True):
        metrics=compute_group(g,args.ece_bins)
        row={"dataset":keys[0],method:keys[1],"seed":int(keys[2]),**metrics}
        rows.append(row)
    seed=pd.DataFrame(rows)
    summary=[]
    metric_cols=["AUC","F1","NLL","Brier","ECE"]
    for keys,g in seed.groupby(["dataset",method],sort=True):
        row={"dataset":keys[0],method:keys[1],"n_seeds":len(g)}
        for m in metric_cols:
            row[f"{m}_mean"]=g[m].mean(); row[f"{m}_sd"]=g[m].std(ddof=1) if len(g)>1 else 0.0
        summary.append(row)
    Path(args.seed_output).parent.mkdir(parents=True,exist_ok=True)
    seed.to_csv(args.seed_output,index=False)
    pd.DataFrame(summary).to_csv(args.summary_output,index=False)
    print(f"Recomputed metrics for {len(seed)} dataset/{method}/seed groups using {args.ece_bins} ECE bins")

if __name__=="__main__": main()
