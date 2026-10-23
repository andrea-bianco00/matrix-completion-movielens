"""Reproducible spectral-boundary probe (not a universal stability guarantee)."""
import json
import numpy as np
from sparse_svp import sparse_svp

cases=[]
for eps in (0.,1e-10,1e-7,1e-4):
    a=np.diag([5.,5.-eps,1.,.2,0.,0.])
    rows,cols=np.indices(a.shape)
    for rank in (1,2):
        out=sparse_svp(a.shape,(rows.ravel(),cols.ravel()),a.ravel(),rank=rank,max_iter=1,tol=0)
        x=out.to_dense()
        singular=np.linalg.svd(a,compute_uv=False)
        optimum=float(np.linalg.norm(singular[rank:]))
        achieved=float(np.linalg.norm(a-x,'fro'))
        cases.append(dict(epsilon=eps,rank=rank,optimum_frobenius=optimum,achieved_frobenius=achieved,gap=achieved-optimum,relative_gap=(achieved-optimum)/max(1.,optimum)))
assert all(abs(c['gap'])<1e-6 for c in cases),cases
with open('../experiment/spectral_probe_14.json','w') as f:json.dump(cases,f,indent=2)
print('Spectral probes:',len(cases),'max abs optimality gap:',max(abs(c['gap']) for c in cases))
