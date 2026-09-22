"""Frozen MiniLM text baseline; identical leave-one-group-out splits to script 13.
Weights must be downloaded locally; no study text is sent to a service.
Reports pooled out-of-fold AUROC (some construction folds are single-class).
"""
import argparse, hashlib, importlib.util, json, sys
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from icaa.schema import load_items
spec = importlib.util.spec_from_file_location('holdout', Path(__file__).with_name('13_group_holdout.py'))
holdout = importlib.util.module_from_spec(spec); spec.loader.exec_module(holdout)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--model-dir', required=True)
    ap.add_argument('--out', default='runs/semantic_baseline.json'); args=ap.parse_args()
    torch.manual_seed(0); torch.set_num_threads(4)
    items=load_items('data/pairs/pilot_v2.jsonl'); text=[i.user_turn for i in items]
    tok=AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    model=AutoModel.from_pretrained(args.model_dir, local_files_only=True).eval()
    batches=[]
    with torch.inference_mode():
        for k in range(0,len(text),32):
            inp=tok(text[k:k+32],padding=True,truncation=True,max_length=256,return_tensors='pt')
            h=model(**inp).last_hidden_state; mask=inp['attention_mask'].unsqueeze(-1)
            pooled=(h*mask).sum(1)/mask.sum(1).clamp(min=1)
            batches.append(torch.nn.functional.normalize(pooled,p=2,dim=1).numpy())
    X=np.concatenate(batches); y=np.array([i.intent for i in items])
    out={'model':'sentence-transformers/all-MiniLM-L6-v2','revision':Path(args.model_dir,'revision.txt').read_text().strip(),
         'data_sha256':hashlib.sha256(Path('data/pairs/pilot_v2.jsonl').read_bytes()).hexdigest(),
         'pooling':'attention-mask mean then L2 normalization','max_length':256,
         'classifier':'train-only StandardScaler + LogisticRegression(C=1, liblinear dual, random_state=0)',
         'metric':'pooled out-of-fold AUROC','n':len(y),'results':{}}
    for name,g in [('seed',[i.pair_id for i in items]),('verb',[holdout.verb_group(t) for t in text]),('domain',[i.domain for i in items])]:
        g=np.array(g); pred=np.full(len(y),np.nan)
        for group in sorted(set(g)):
            te=g==group; tr=~te
            if len(np.unique(y[tr]))!=2: raise ValueError('single-class training fold')
            sc=StandardScaler().fit(X[tr]); clf=LogisticRegression(C=1,solver='liblinear',dual=True,max_iter=5000,random_state=0)
            clf.fit(sc.transform(X[tr]),y[tr]); pred[te]=clf.decision_function(sc.transform(X[te]))
        assert np.isfinite(pred).all()
        out['results'][name]={'auroc':float(roc_auc_score(y,pred)),'n_groups':len(set(g)),'scores':pred.tolist()}
        print(name, out['results'][name]['auroc'], flush=True)
    out['item_ids']=[i.item_id for i in items]; out['labels']=y.tolist()
    Path(args.out).write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__': main()
