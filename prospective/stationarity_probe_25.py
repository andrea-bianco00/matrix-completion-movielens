"""Training-only SVT optimality probe; no validation/test outcomes accessed."""
import csv, hashlib, io, json, os, sys, time, zipfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'revised'))
from scalable_19 import svt_matrix_free
from stationarity_25 import matrix_free_pg_residual
source=Path('/mnt/data/Dimension Reduction and Matrix Completion.zip')
manifest=json.loads((ROOT/'prospective/split_manifest_18.json').read_text())
with zipfile.ZipFile(source) as z:
    names=[n for n in z.namelist() if n.endswith('/ratings.csv')]
    assert len(names)==1
    raw=z.read(names[0])
assert hashlib.sha256(raw).hexdigest()==manifest['source_sha256']
records=list(csv.DictReader(io.StringIO(raw.decode())))
users=sorted({int(r['userId']) for r in records}); movies=sorted({int(r['movieId']) for r in records})
um={v:i for i,v in enumerate(users)}; mm={v:i for i,v in enumerate(movies)}
with np.load(ROOT/'prospective/split_indices_18.npz') as split: train=split['train'].copy()
assert hashlib.sha256(train.astype('<i8').tobytes()).hexdigest()==manifest['index_sha256']['train']
tr=[records[int(i)] for i in train]
rows=np.array([um[int(r['userId'])] for r in tr]); cols=np.array([mm[int(r['movieId'])] for r in tr]); vals=np.array([float(r['rating']) for r in tr])
shape=(len(users),len(movies))
lam=float(sys.argv[1]); budget=int(sys.argv[2]); start=time.perf_counter()
fit=svt_matrix_free(shape,(rows,cols),vals,threshold=lam,max_iter=budget,initial_rank=8)
stationarity=matrix_free_pg_residual(fit,shape,(rows,cols),vals,threshold=lam,initial_rank=8)
result={'lambda':lam,'iteration_budget':budget,'iterations':len(fit.history),
        'stop_reason':fit.stop_reason,'train_rmse':fit.history[-1],
        'fitted_rank':len(fit.singular_values),'stationarity':stationarity,
        'elapsed_seconds':time.perf_counter()-start,
        'source_sha256':manifest['source_sha256'],
        'train_index_sha256':manifest['index_sha256']['train']}
print(json.dumps(result,indent=2))
