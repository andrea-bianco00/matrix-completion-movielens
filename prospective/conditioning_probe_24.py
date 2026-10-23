"""Training-only resource feasibility; no validation/test outcomes read."""
import csv,hashlib,io,json,os,resource,sys,time,zipfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'revised'))
from scalable_19 import svt_matrix_free,admira_matrix_free
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
kind=sys.argv[1]; arg=float(sys.argv[2]); iters=int(sys.argv[3]); shape=(len(users),len(movies))
t0=time.perf_counter(); out={'algorithm':kind,'parameter':arg,'iterations_budget':iters,'train_n':len(vals),'shape':shape,'status':'unknown'}
try:
    if kind=='SVT': fit=svt_matrix_free(shape,(rows,cols),vals,threshold=arg,step=1.,max_iter=iters,initial_rank=8)
    elif kind=='ADMiRA':
        diagnostics=[]
        fit=admira_matrix_free(shape,(rows,cols),vals,rank=int(arg),max_iter=iters,max_design_bytes=256_000_000,conditioning_log=diagnostics)
        out['conditioning']=diagnostics
    else: raise ValueError('kind')
    out.update(status='ok',rank=len(fit.singular_values),history=fit.history,stop_reason=fit.stop_reason)
except Exception as exc: out.update(status='error',exception_type=type(exc).__name__,exception=str(exc))
out['wall_seconds']=time.perf_counter()-t0
out['peak_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
print(json.dumps(out,indent=2))
