"""Training-only first-order diagnostic on frozen MovieLens split. No validation/test labels."""
import csv,hashlib,io,json,os,sys,time,zipfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'revised'))
from scalable_19 import admira_matrix_free
from sparse_stationarity_27 import tangent_gradient_sparse
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
rank=int(sys.argv[1]); iters=int(sys.argv[2]); shape=(len(users),len(movies))
t0=time.perf_counter()
fit=admira_matrix_free(shape,(rows,cols),vals,rank=rank,max_iter=iters,max_design_bytes=256_000_000)
diag=tangent_gradient_sparse(fit,rows,cols,vals,shape,rank)
output={'rank':rank,'iterations_budget':iters,'iterations_run':len(fit.history),'stop_reason':fit.stop_reason,'train_n':len(vals),'shape':shape,'train_rmse':fit.history[-1],**diag,'wall_seconds':time.perf_counter()-t0}
print(json.dumps(output,indent=2))
