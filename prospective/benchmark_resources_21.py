"""Training-only feasibility probe; never evaluates validation or test labels."""
import csv, hashlib, io, json, os, resource, sys, time, zipfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'revised'))
from scalable_19 import svt_matrix_free, admira_matrix_free
SOURCE=Path('/mnt/data/Dimension Reduction and Matrix Completion.zip')
with zipfile.ZipFile(SOURCE) as z:
    names=[p for p in z.namelist() if p.endswith('/ratings.csv')]
    if len(names)!=1: raise ValueError('ratings.csv ambiguous')
    raw=z.read(names[0])
manifest=json.loads((ROOT/'prospective'/'split_manifest_18.json').read_text())
assert hashlib.sha256(raw).hexdigest()==manifest['source_sha256']
records=list(csv.DictReader(io.StringIO(raw.decode())))
users=sorted({int(x['userId']) for x in records}); movies=sorted({int(x['movieId']) for x in records})
um={x:i for i,x in enumerate(users)}; mm={x:i for i,x in enumerate(movies)}
with np.load(ROOT/'prospective'/'split_indices_18.npz') as parts:
    train=parts['train'].copy()
assert hashlib.sha256(train.astype('<i8').tobytes()).hexdigest()==manifest['index_sha256']['train']
# Only training rows are materialized into numeric arrays. Held-out rating labels are not read.
tr=[records[int(i)] for i in train]
rows=np.array([um[int(x['userId'])] for x in tr],dtype=int)
cols=np.array([mm[int(x['movieId'])] for x in tr],dtype=int)
vals=np.array([float(x['rating']) for x in tr],dtype=float)
shape=(len(users),len(movies)); kind=sys.argv[1]
start=time.perf_counter()
out={'algorithm':kind,'shape':shape,'training_observations':len(vals),'iterations_budget':1,'status':'unknown'}
try:
    if kind=='SVT':
        fit=svt_matrix_free(shape,(rows,cols),vals,threshold=8.,step=1.,max_iter=1,initial_rank=8)
    elif kind=='ADMiRA':
        fit=admira_matrix_free(shape,(rows,cols),vals,rank=2,max_iter=1,max_design_bytes=256_000_000)
    else: raise ValueError('unknown algorithm')
    out.update(status='ok',train_rmse=fit.history[-1],rank=len(fit.singular_values),stop_reason=fit.stop_reason)
except Exception as exc:
    out.update(status='error',exception_type=type(exc).__name__,exception=str(exc))
out['wall_seconds']=time.perf_counter()-start
out['maxrss_kib_linux']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
print(json.dumps(out,indent=2),flush=True)
