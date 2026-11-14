"""P5 validation-only, ablations using the final shared best-F2 threshold; never loads test."""
import sys,json,hashlib,platform,subprocess
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'model'))
import tier2_autoencoder as m
import numpy as np
import pandas as pd
import sklearn,torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=ROOT/'results'/'p5'/'ablation'
OUT.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(1)
args=SimpleNamespace(hidden=[16,8],epochs=200,batch_size=64,lr=.001,weight_decay=1e-5,threshold="best_f2")
Xtr,_=m.load_split('train',use_normal_only=True)
Xva,y=m.load_split('val')
prep=m.fit_preprocess(Xtr)
rows=[];scores=[]
for fs in ['ALL21','NO_VARIABILITY17','DROP_WIDTH20','PRUNED18']:
 cols=m.feature_columns(fs); idx=[m.FEATURES.index(c) for c in cols]
 tr=prep(Xtr)[:,idx]; va=prep(Xva)[:,idx]
 for seed in range(5):
  model=m.train_ae(tr,args,seed)
  trh=m.reconstruct(model,tr); vah=m.reconstruct(model,va)
  for kind in (['mahalanobis','mse','cosine'] if fs=='DROP_WIDTH20' else ['mahalanobis']):
   s=m.Scorer(kind,'empirical').fit(tr,trh)(va,vah)
   tau=m.choose_tau(args,None,s,y)
   measured=m.metrics(y,s,tau)
   row=dict(feature_set=fs,score=kind,seed=seed,tau=tau,ap=measured.pop('pr_auc'),**measured)
   rows.append(row)
   scores.append(pd.DataFrame(dict(row_id=Xva.index,seed=seed,feature_set=fs,scoring=kind,split='val',label=y,score=s,tau=tau,pred=(s>=tau).astype(int))))
   print(row,flush=True)
df=pd.DataFrame(rows);df.to_csv(OUT/'metrics_per_seed.csv',index=False)
pd.concat(scores).to_csv(OUT/'validation_scores.csv',index=False)
summary=df.groupby(['feature_set','score'])[['ap','roc_auc','f1','f2','precision','recall','fpr','n_alerts']].agg(['mean','std'])
summary.to_csv(OUT/'summary.csv')
for table,sub,base in [('features',df[df.score=='mahalanobis'],'ALL21'),('scoring',df[df.feature_set=='DROP_WIDTH20'],'mahalanobis')]:
 key='feature_set' if table=='features' else 'score'
 b=sub[sub[key]==base].set_index('seed')
 records=[]
 for name,g in sub.groupby(key,sort=False):
  g=g.set_index('seed');d=g.ap-b.ap
  records.append(dict(setting=name,ap_mean=g.ap.mean(),ap_std=g.ap.std(),roc_auc_mean=g.roc_auc.mean(),roc_auc_std=g.roc_auc.std(),delta_ap=d.mean(),paired_delta_std=d.std(),positive_seeds=int((d>0).sum()),**{k+'_mean':g[k].mean() for k in ['f1','f2','precision','recall','fpr','n_alerts']},**{k+'_std':g[k].std() for k in ['f1','f2','precision','recall','fpr','n_alerts']}))
 t=pd.DataFrame(records);t.to_csv(OUT/(table+'_comparison.csv'),index=False)
 fig,ax=plt.subplots(figsize=(7,3.7)); x=np.arange(len(t))
 ax.errorbar(x,t.ap_mean,yerr=t.ap_std,fmt='o',capsize=5,label='Mean ± seed SD')
 for i,name in enumerate(t.setting):ax.scatter(np.repeat(i,5),sub[sub[key]==name].ap,s=16,alpha=.5)
 ax.set_xticks(x,t.setting);ax.set_ylabel('Validation average precision (AP)');ax.set_ylim(0,1);ax.legend();fig.tight_layout()
 fig.savefig(OUT/(table+'.png'),dpi=180);fig.savefig(OUT/(table+'.pdf'));plt.close(fig)
meta=dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__,training=vars(args),seeds=list(range(5)),cov='empirical + 1e-6 I',clip=0,threshold='best_f2; final shared protocol',evaluation_split='val',train_normal_count=len(Xtr),val_count=len(Xva),val_anomalies=int(y.sum()),split_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'data/splits/train.csv',ROOT/'data/splits/val.csv']})
(OUT/'provenance.json').write_text(json.dumps(meta,indent=2))
print(summary)
