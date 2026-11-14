"""Analyze P3's published frozen scores. No fitting or test-set tuning."""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
from sklearn.metrics import average_precision_score,roc_auc_score,fbeta_score,precision_score,recall_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1];SRC=R/'results/tier2/final_mahalanobis_best_f2_DROP_WIDTH20';O=R/'results/p5_analysis';O.mkdir(exist_ok=True)
s=pd.read_csv(SRC/'scores.csv');s=s[s.split=='test'].copy();raw=pd.read_csv(R/'data/splits/test.csv');train=pd.read_csv(R/'data/splits/train.csv');normal=train[train.label==0]
e=pd.read_csv(SRC/'feature_errors.csv');e=e[e.split=='test'];cols=[x for x in e if x not in ['row_id','seed','split']]
assert len(s)==378*5 and not s.duplicated(['row_id','seed']).any()
assert (s.pred==(s.score>=s.tau).astype(int)).all()
s=s.merge(raw,on=['row_id','label'],validate='many_to_one');s['margin']=s.score-s.tau;s['ratio']=s.score/s.tau
s['outcome']=np.select([(s.label==1)&(s.pred==1),(s.label==0)&(s.pred==1),(s.label==1)&(s.pred==0)],['TP','FP','FN'],default='TN')
rows=[]
for seed,g in s.groupby('seed'):
 counts=g.outcome.value_counts();r=dict(seed=int(seed),tau=g.tau.iloc[0],**{k:int(counts.get(k,0)) for k in ['TP','FP','FN','TN']},ap=average_precision_score(g.label,g.score),roc_auc=roc_auc_score(g.label,g.score),f2=fbeta_score(g.label,g.pred,beta=2),precision=precision_score(g.label,g.pred),recall=recall_score(g.label,g.pred))
 rows.append(r)
metrics=pd.DataFrame(rows);metrics.to_csv(O/'test_per_seed.csv',index=False)
pub=pd.read_csv(SRC/'metrics.csv');pub=pub[pub.split=='test'].set_index('seed')
for x in metrics.itertuples():
 for k,p in [('ap','pr_auc'),('roc_auc','roc_auc'),('f2','f2'),('precision','precision'),('recall','recall')]:assert np.isclose(getattr(x,k),pub.loc[x.seed,p])
a=s.groupby(['row_id','exam','label']).agg(alert_seeds=('pred','sum'),score_min=('score','min'),score_max=('score','max'),ratio_min=('ratio','min'),ratio_max=('ratio','max')).reset_index();a['error_seeds']=np.where(a.label==1,5-a.alert_seeds,a.alert_seeds);a.to_csv(O/'test_sample_stability.csv',index=False)
# Exhaustively include all five-seed errors; no preferred model seed.
caseids=sorted(a[a.error_seeds==5].row_id.tolist())
assert any(a[(a.row_id.isin(caseids))].label==0) and any(a[(a.row_id.isin(caseids))].label==1)
# Diagnostic alert: largest minimum score/tau among five-seed true positives; tie by row_id.
tp=a[(a.label==1)&(a.alert_seeds==5)].sort_values(['ratio_min','row_id'],ascending=[False,True]);flag=int(tp.iloc[0].row_id)
case=s[s.row_id.isin(caseids+[flag])].copy();case.to_csv(O/'case_details_all_seeds.csv',index=False)
case[['row_id','exam','label']+list(raw.columns[3:])].drop_duplicates().to_csv(O/'case_raw_features.csv',index=False)
diag=[]
for rid in caseids+[flag]:
 errs=e[e.row_id==rid]
 for seed,g in errs.groupby('seed'):
  for rank,(feature,val) in enumerate(g.iloc[0][cols].astype(float).sort_values(ascending=False).items(),1):
   value=float(raw.set_index('row_id').loc[rid,feature]);ref=normal[feature]
   diag.append(dict(row_id=rid,seed=seed,feature=feature,rank=rank,squared_standardized_residual=val,raw_value=value,normal_train_median=ref.median(),normal_train_p05=ref.quantile(.05),normal_train_p95=ref.quantile(.95),normal_train_percentile=(ref<=value).mean()*100))
d=pd.DataFrame(diag);d.to_csv(O/'reconstruction_diagnostics.csv',index=False)
sub=[]
for seed,g in s[s.label==1].groupby('seed'):
 for name,z in g.groupby(g.DP.gt(0)):
  sub.append(dict(seed=seed,group='DP_present' if name else 'DP_absent',n=len(z),exams=z.exam.nunique(),detected=int(z.pred.sum()),recall=z.pred.mean()))
pd.DataFrame(sub).to_csv(O/'subgroup_recall.csv',index=False)
tc=pd.read_csv(SRC/'threshold_comparison.csv');tc[tc.split=='test'].to_csv(O/'test_threshold_comparison.csv',index=False)
fig,ax=plt.subplots(figsize=(7,4))
for rid,g in case.groupby('row_id'):ax.plot(g.seed,g.ratio,'o-',label=f'{rid} ({g.outcome.iloc[0]})')
ax.axhline(1,color='black',ls='--',label='Alert threshold');ax.set(xlabel='Training seed',ylabel='Score / threshold',xticks=range(5));ax.legend(ncol=2);fig.tight_layout();fig.savefig(O/'case_margins.png',dpi=180);fig.savefig(O/'case_margins.pdf');plt.close(fig)
top=d[d.row_id==flag].groupby('feature').squared_standardized_residual.agg(['mean','std']).sort_values('mean',ascending=False).head(6)
fig,ax=plt.subplots(figsize=(6,3.5));ax.bar(top.index,top['mean'],yerr=top['std'],capsize=3);ax.set_ylabel('Squared standardized residual (mean ± seed SD)');ax.set_title(f'Flagged true-positive row {flag}: reconstruction diagnostic');fig.tight_layout();fig.savefig(O/'flagged_diagnostic.png',dpi=180);fig.savefig(O/'flagged_diagnostic.pdf');plt.close(fig)
meta=dict(source=str(SRC.relative_to(R)),rule='validation best_f2; published tau and predictions used unchanged',case_selection='All test rows misclassified in all five seeds; flagged TP maximizes minimum score/tau among five-seed TPs, tie by row_id. Descriptive post-hoc selection; not representative sampling.',error_case_ids=caseids,flagged_id=flag,test_model_retrained=False,source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in SRC.glob('*.csv')});(O/'provenance.json').write_text(json.dumps(meta,indent=2))
print(metrics.to_string(index=False));print('CASES',caseids,'FLAG',flag);print(a[a.row_id.isin(caseids+[flag])].to_string(index=False));print(top)
