"""Reproduce two already-selected validation scores with isolated fit input."""
import csv, json, tempfile
from pathlib import Path
import numpy as np
from isolated_runner_37 import run_child
from validation_32 import score_validation
from isolation_36 import sha
ROOT=Path(__file__).resolve().parent.parent
EXP=ROOT/'isolated_export_36'
manifest=json.loads((EXP/'export_manifest.json').read_text())
vp=EXP/'validation'/'validation.csv'
if sha(vp)!=manifest['partitions']['validation']['sha256']: raise RuntimeError('validation checksum mismatch')
cat=json.loads((EXP/'fit'/'catalog.json').read_text())
ui={u:i for i,u in enumerate(cat['users'])};mi={m:i for i,m in enumerate(cat['items'])}
with vp.open(newline='') as f: data=list(csv.DictReader(f))
vr=np.array([ui[int(z['userId'])] for z in data],dtype=np.int64)
vc=np.array([mi[int(z['movieId'])] for z in data],dtype=np.int64)
y=np.array([float(z['rating']) for z in data])
with (EXP/'fit'/'train.csv').open(newline='') as f: train=list(csv.DictReader(f))
warm=np.isin(vc,np.array([mi[int(z['movieId'])] for z in train]))
old=json.loads((ROOT/'prospective'/'validation_results_34.json').read_text())['records']
records=[]
with tempfile.TemporaryDirectory() as d:
    q=Path(d)/'coordinates.npz';np.savez(q,rows=vr,cols=vc)
    for fam,params in [('global_mean',{}),('user_item_bias',{'regularization':5.0})]:
        res=run_child(fam,params,EXP/'fit',q,seconds=120)
        if res['status']!='ok': raise RuntimeError(str(res))
        metric=score_validation(res.pop('predictions'),y,warm)
        original=next(z for z in old if z['family']==fam and z['params']==params)
        delta=abs(metric['validation_rmse']-original['validation_rmse'])
        records.append({'family':fam,'params':params,'isolated_validation_rmse':metric['validation_rmse'],
                        'previous_validation_rmse':original['validation_rmse'],'absolute_delta':delta,
                        'training_rmse':res['training_rmse'],'status':res['status'],
                        'warm_item_n':metric['warm_item_n'],'cold_item_n':metric['cold_item_n']})
        if delta>1e-10: raise AssertionError('score changed')
print(json.dumps(records,indent=2))
(ROOT/'prospective'/'isolated_equivalence_37.json').write_text(json.dumps(records,indent=2)+'\n')
