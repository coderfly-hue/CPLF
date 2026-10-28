"""Regression tests for manuscript-aligned canonical artifact rules."""
from __future__ import annotations
import csv, subprocess, sys, tempfile, unittest
from pathlib import Path
import pandas as pd
from scripts.verify_canonical_release import check_feedback

REPO=Path(__file__).resolve().parents[1]
SCRIPTS=REPO/'scripts'
FORMAL=REPO/'CPLF LLM protocols io complete package'/'CPLF LLM protocols io FORMAL RELEASE'

class CanonicalEvidenceTests(unittest.TestCase):
    def test_manuscript_aligned_modules_exist(self):
        expected=['data.py','profile.py','memory.py','model.py','prediction.py','retrieval.py','planning.py','generation.py','verification.py','pipeline.py']
        for name in expected:
            self.assertTrue((REPO/'src/cplf_llm'/name).is_file(),name)
        for i in range(1,9): self.assertTrue((REPO/f'paper/figures/figure{i}.png').is_file())

    def test_feedback_raw_scores_are_preserved_and_normalized(self):
        with tempfile.TemporaryDirectory(dir=REPO.parent) as td:
            src=Path(td)/'in.csv'; out=Path(td)/'out.csv'
            pd.DataFrame([{
                'dataset':'MathDial','case_id':'c1','method':'CPLF-LLM','seed':2024,'relevance':0.8,
                'diagnostic_raw':4,'actionability_raw':5,'personalization_raw':3,'leakage_raw':1,
                'feedback_final':'Review the relevant concept, then check your next step.',
                'evaluator_raw_response':'{"diagnostic":4,"actionability":5,"personalization":3,"leakage":1}'
            }]).to_csv(src,index=False)
            subprocess.run([sys.executable,str(SCRIPTS/'canonicalize_feedback_outputs.py'),'--input',str(src),'--output',str(out)],check=True)
            d=pd.read_csv(out).iloc[0]
            self.assertEqual(d.diagnostic_consistency,0.75)
            self.assertEqual(d.actionability,1.0)
            self.assertEqual(d.personalization,0.5)
            self.assertEqual(d.answer_leakage,0.0)
            self.assertAlmostEqual(d.fqs,0.7625)
            errors=[]
            self.assertEqual(check_feedback(out,errors),1)
            self.assertEqual(errors,[])

    def test_continuous_legacy_rubric_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=REPO.parent) as td:
            src=Path(td)/'in.csv'; out=Path(td)/'out.csv'
            pd.DataFrame([{
                'dataset':'MathDial','case_id':'c1','method':'CPLF-LLM','seed':2024,'relevance':0.8,
                'diagnostic_consistency':0.85504,'actionability':0.75,'personalization':0.5,'answer_leakage':0.25
            }]).to_csv(src,index=False)
            p=subprocess.run([sys.executable,str(SCRIPTS/'canonicalize_feedback_outputs.py'),'--input',str(src),'--output',str(out)],capture_output=True,text=True)
            self.assertNotEqual(p.returncode,0)
            self.assertIn('do not round legacy scores',p.stderr+p.stdout)

    def test_normalized_only_scores_are_not_relabelled_as_original_evaluator_output(self):
        with tempfile.TemporaryDirectory(dir=REPO.parent) as td:
            src=Path(td)/'in.csv'; out=Path(td)/'out.csv'
            pd.DataFrame([{
                'dataset':'MathDial','case_id':'c1','method':'CPLF-LLM','seed':2024,
                'relevance':0.8,'diagnostic_consistency':0.75,'actionability':1.0,
                'personalization':0.5,'answer_leakage':0.0,
                'feedback_final':'Useful genuine feedback.',
                'evaluator_raw_response':'{"diagnostic":4,"actionability":5,"personalization":3,"leakage":1}'
            }]).to_csv(src,index=False)
            p=subprocess.run([sys.executable,str(SCRIPTS/'canonicalize_feedback_outputs.py'),
                              '--input',str(src),'--output',str(out)],capture_output=True,text=True)
            self.assertNotEqual(p.returncode,0)
            self.assertIn('Missing original evaluator integer field',p.stderr+p.stdout)

    def test_proxy_ablation_is_rejected(self):
        legacy=FORMAL/'per_case_outputs/ablation_outputs.csv'
        with tempfile.TemporaryDirectory(dir=REPO.parent) as td:
            p=subprocess.run([sys.executable,str(SCRIPTS/'canonicalize_ablation_outputs.py'),'--input',str(legacy),'--output',str(Path(td)/'out.csv')],capture_output=True,text=True)
            self.assertNotEqual(p.returncode,0)
            self.assertIn('Proxy columns',p.stderr+p.stdout)

    def test_prediction_metrics_recomputed_from_probabilities(self):
        with tempfile.TemporaryDirectory(dir=REPO.parent) as td:
            src=Path(td)/'p.csv'; seed=Path(td)/'seed.csv'; summary=Path(td)/'sum.csv'
            pd.DataFrame([
                {'dataset':'D','method':'M','seed':2024,'y_true':0,'y_prob':0.1},
                {'dataset':'D','method':'M','seed':2024,'y_true':1,'y_prob':0.9},
                {'dataset':'D','method':'M','seed':2024,'y_true':0,'y_prob':0.2},
                {'dataset':'D','method':'M','seed':2024,'y_true':1,'y_prob':0.8},
            ]).to_csv(src,index=False)
            subprocess.run([sys.executable,str(SCRIPTS/'recompute_prediction_metrics.py'),'--input',str(src),'--seed-output',str(seed),'--summary-output',str(summary),'--ece-bins','10'],check=True)
            d=pd.read_csv(seed).iloc[0]
            self.assertAlmostEqual(d.AUC,1.0)
            self.assertAlmostEqual(d.F1,1.0)

    def test_release_config_records_implementation_specific_ece_and_word_limit(self):
        text=(FORMAL/'protocol_config.yaml').read_text(encoding='utf-8')
        self.assertIn('ece_bins: 10',text)
        self.assertIn('max_feedback_words: 150',text)

if __name__=='__main__': unittest.main()
