"""Equivalence audit of remaining nine candidates; validation used solely to compare frozen scores."""
import csv,json,tempfile,sys
from pathlib import Path
import numpy as np
from isolated_runner_37 import run_child
from validation_32 import score_validation
from isolation_36 import sha
ROOT=Path(__file__).resolve().parent.parent
EXP=ROOT/'isolated_export_36'
manifest=json.loads((EXP/'export_manifest.json').read_text())
vp=EXP/'validation'/'validation.csv'
assert sha(vp)==manifest['partitions']['validation']['sha256']
cat=json.loads((EXP/'fit'/'catalog.json').read_text())
ui={int(u):i for i,u in enumerate(cat['users'])}; mi={int(m):i for i,m in enumerate(cat['items'])}
with vp.open(newline='') as f: data=list(csv.DictReader(f))
vr=np.array([ui[int(z['userId'])] for z in data],dtype=np.int64)
vc=np.array([mi[int(z['movieId'])] for z in data],dtype=np.int64)
y=np.array([float(z['rating']) for z in data])
with (EXP/'fit'/'train.csv').open(newline='') as f: train=list(csv.DictReader(f))
warm=np.isin(vc,np.array([mi[int(z['movieId'])] for z in train]))
old=json.loads((ROOT/'prospective'/'validation_results_34.json').read_text())['records']
out=ROOT/'prospective'/'isolated_equivalence_remaining_40.json'
records=json.loads(out.read_text()) if out.exists() else []
with tempfile.TemporaryDirectory() as d:
    q=Path(d)/'coordinates.npz';np.savez(q,rows=vr,cols=vc)
    for original in old:
        fam,params=original['family'],original['params']
        if fam=='global_mean' or (fam=='user_item_bias' and params.get('regularization')==5.) or any(z['family']==fam and z['params']==params for z in records):continue
        print('RUN',fam,params,flush=True)
        res=run_child(fam,params,EXP/'fit',q,seconds=120)
        record={'family':fam,'params':params,'status':res['status'],'previous_validation_rmse':original['validation_rmse']}
        if res['status']=='ok':
            metric=score_validation(res.pop('predictions'),y,warm)
            delta=abs(metric['validation_rmse']-original['validation_rmse'])
            record.update(isolated_validation_rmse=metric['validation_rmse'],absolute_delta=delta,training_rmse=res['training_rmse'],iterations=res['iterations'],stop_reason=res['stop_reason'],wall_seconds=res['wall_seconds'])
            record['equivalent']=bool(delta<=1e-8)
        else:record['error']=res.get('stderr_tail','')
        records.append(record)
        (ROOT/'prospective'/'isolated_equivalence_remaining_40.json').write_text(json.dumps(records,indent=2)+'\n')
        print(json.dumps(record),flush=True)
if len(records)!=9 or not all(z.get('equivalent') for z in records):sys.exit(2)
