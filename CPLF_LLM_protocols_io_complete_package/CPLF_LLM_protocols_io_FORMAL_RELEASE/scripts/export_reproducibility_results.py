#!/usr/bin/env python3
"""Build public statistical/seed-level exports from the ACTUAL archived CPLF-LLM outputs.

This utility does not reconstruct missing runs from manuscript means. It is intended
for use immediately before public release, when the final held-out prediction,
feedback, ablation, and per-seed files are available.
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, log_loss, brier_score_loss


def metric(rows: pd.DataFrame, name: str) -> float:
    if name == 'auc':
        y=rows['y_true'].to_numpy(int); p=rows['y_prob'].to_numpy(float)
        if len(np.unique(y)) < 2: return np.nan
        return float(roc_auc_score(y,p))
    if name == 'f1': return float(f1_score(rows['y_true'], rows['y_prob']>=0.5))
    if name == 'nll': return float(log_loss(rows['y_true'], rows['y_prob'], labels=[0,1]))
    if name == 'brier': return float(brier_score_loss(rows['y_true'], rows['y_prob']))
    if name in rows.columns: return float(pd.to_numeric(rows[name], errors='coerce').mean())
    raise ValueError(name)


def paired_bootstrap(df, ref, other, metric_name, cluster, n_boot=10000, seed=2024):
    a=df[df.method==ref].copy(); b=df[df.method==other].copy()
    shared=sorted(set(a[cluster]) & set(b[cluster]))
    if not shared: raise ValueError(f'No shared {cluster} values for {ref} and {other}')
    ag={k:g for k,g in a[a[cluster].isin(shared)].groupby(cluster)}
    bg={k:g for k,g in b[b[cluster].isin(shared)].groupby(cluster)}
    observed=metric(pd.concat([ag[k] for k in shared]), metric_name)-metric(pd.concat([bg[k] for k in shared]), metric_name)
    rng=np.random.default_rng(seed); diffs=[]; attempts=0
    while len(diffs)<n_boot and attempts<n_boot*5:
        attempts+=1; samp=rng.choice(shared, len(shared), replace=True)
        aa=pd.concat([ag[k] for k in samp], ignore_index=True); bb=pd.concat([bg[k] for k in samp], ignore_index=True)
        da=metric(aa,metric_name); db=metric(bb,metric_name)
        if np.isfinite(da) and np.isfinite(db): diffs.append(da-db)
    if len(diffs)<n_boot: raise RuntimeError('Too many invalid bootstrap resamples; inspect outcome balance.')
    d=np.asarray(diffs); lo,hi=np.percentile(d,[2.5,97.5])
    p=2*min((np.sum(d<=0)+1)/(len(d)+1),(np.sum(d>=0)+1)/(len(d)+1)); p=min(1.0,p)
    return dict(reference=ref,comparison=other,metric=metric_name,difference=observed,ci_low=lo,ci_high=hi,p_raw=p,n_boot=n_boot,cluster=cluster)


def holm(p):
    p=np.asarray(p,float); m=len(p); order=np.argsort(p); out=np.zeros(m); running=0
    for rank,idx in enumerate(order):
        running=max(running,(m-rank)*p[idx]); out[idx]=min(1,running)
    return out


def cmd_pairwise(a):
    df=pd.read_csv(a.input)
    methods=[x for x in sorted(df.method.unique()) if x!=a.reference]
    if a.methods: methods=[x.strip() for x in a.methods.split(',') if x.strip()]
    rows=[paired_bootstrap(df,a.reference,m,a.metric,a.cluster,a.n_boot,a.seed) for m in methods]
    out=pd.DataFrame(rows); out['p_holm']=holm(out.p_raw); out.to_csv(a.output,index=False)


def cmd_ablation(a):
    df=pd.read_csv(a.input)
    if 'variant' in df.columns and 'method' not in df.columns: df=df.rename(columns={'variant':'method'})
    methods=[x for x in sorted(df.method.unique()) if x!=a.reference]
    rows=[paired_bootstrap(df,a.reference,m,a.metric,a.cluster,a.n_boot,a.seed) for m in methods]
    out=pd.DataFrame(rows); out['p_holm']=holm(out.p_raw); out.to_csv(a.output,index=False)


def cmd_collect(a):
    root=Path(a.root); rows=[]
    for p in root.rglob('test_metrics_seed_*.json'):
        obj=json.loads(p.read_text(encoding='utf-8')); m=re.search(r'seed_(\d+)',p.name); seed=int(m.group(1)) if m else obj.get('seed')
        rows.append({'experiment':'prediction','dataset':p.parent.name,'method':a.method,'seed':seed,**obj})
    if not rows: raise SystemExit('No test_metrics_seed_*.json files found. Copy actual run logs into the output tree first.')
    pd.DataFrame(rows).to_csv(a.output,index=False)


def main():
    p=argparse.ArgumentParser(); s=p.add_subparsers(dest='cmd',required=True)
    q=s.add_parser('pairwise'); q.add_argument('--input',required=True); q.add_argument('--reference',default='CPLF-LLM'); q.add_argument('--methods'); q.add_argument('--metric',required=True); q.add_argument('--cluster',required=True); q.add_argument('--n-boot',type=int,default=10000); q.add_argument('--seed',type=int,default=2024); q.add_argument('--output',required=True); q.set_defaults(func=cmd_pairwise)
    q=s.add_parser('ablation'); q.add_argument('--input',required=True); q.add_argument('--reference',default='CPLF-LLM'); q.add_argument('--metric',required=True); q.add_argument('--cluster',required=True); q.add_argument('--n-boot',type=int,default=10000); q.add_argument('--seed',type=int,default=2024); q.add_argument('--output',required=True); q.set_defaults(func=cmd_ablation)
    q=s.add_parser('collect-seeds'); q.add_argument('--root',required=True); q.add_argument('--method',default='CPLF-LLM'); q.add_argument('--output',required=True); q.set_defaults(func=cmd_collect)
    a=p.parse_args(); a.func(a)
if __name__=='__main__': main()
