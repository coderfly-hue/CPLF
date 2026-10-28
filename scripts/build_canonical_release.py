#!/usr/bin/env python3
"""Orchestrate canonical artifact export from genuine rerun outputs."""
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path
import pandas as pd

def run(cmd):
    print('+',' '.join(map(str,cmd)),flush=True); subprocess.run(cmd,check=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--repo-root',default='.')
    ap.add_argument('--feedback')
    ap.add_argument('--predictions')
    ap.add_argument('--split-assignments',help='Original, canonical-ID fixed split assignments for the supplied predictions')
    ap.add_argument('--ablation-predictions')
    args=ap.parse_args()
    root=Path(args.repo_root).resolve(); scripts=root/'scripts'; can=root/'artifacts'/'canonical'; can.mkdir(parents=True,exist_ok=True)
    py=sys.executable
    if args.feedback:
        run([py,str(scripts/'canonicalize_feedback_outputs.py'),'--input',args.feedback,'--output',str(can/'feedback_outputs.csv')])
    if args.predictions:
        if not args.split_assignments:
            raise SystemExit('Canonical predictions require --split-assignments from the same standardized source; legacy IDs cannot be copied')
        src=Path(args.predictions)
        predictions=pd.read_csv(src)
        splits=pd.read_csv(args.split_assignments)
        required_pred={'dataset','learner_id','case_id','method','seed','y_true','y_prob'}
        required_split={'dataset','learner_or_session_id','split'}
        if required_pred-set(predictions) or required_split-set(splits):
            raise SystemExit('Missing canonical prediction or split-assignment columns')
        if predictions.empty or splits.empty or predictions[list(required_pred)].isna().any().any():
            raise SystemExit('Canonical prediction and split inputs must be nonempty and complete')
        if not predictions.learner_id.astype(str).str.startswith('CID_').all():
            raise SystemExit('Predictions must use canonical CID_ IDs; legacy ANONLEARNER IDs are not canonical')
        if splits.duplicated(['dataset','learner_or_session_id']).any():
            raise SystemExit('Canonical split assignments contain duplicate IDs')
        test_ids=set(zip(splits.loc[splits.split=='test','dataset'],splits.loc[splits.split=='test','learner_or_session_id']))
        pred_ids=set(zip(predictions.dataset,predictions.learner_id))
        if not pred_ids.issubset(test_ids):
            raise SystemExit(f'Prediction IDs absent from canonical test split: {len(pred_ids-test_ids)}')
        (can/'split_assignments.csv').write_bytes(Path(args.split_assignments).read_bytes())
        dst=can/'prediction_outputs.csv'; dst.write_bytes(src.read_bytes())
        run([py,str(scripts/'recompute_prediction_metrics.py'),'--input',str(dst),'--seed-output',str(can/'prediction_seed_metrics.csv'),'--summary-output',str(can/'prediction_summary.csv'),'--ece-bins','10'])
    if args.ablation_predictions:
        run([py,str(scripts/'canonicalize_ablation_outputs.py'),'--input',args.ablation_predictions,'--output',str(can/'ablation_prediction_outputs.csv')])
        run([py,str(scripts/'recompute_prediction_metrics.py'),'--input',str(can/'ablation_prediction_outputs.csv'),'--seed-output',str(can/'ablation_seed_metrics.csv'),'--summary-output',str(can/'ablation_prediction_summary.csv'),'--ece-bins','10','--group-method-column','variant'])
    run([py,str(scripts/'verify_canonical_release.py'),'--root',str(root)])
if __name__=='__main__': main()
