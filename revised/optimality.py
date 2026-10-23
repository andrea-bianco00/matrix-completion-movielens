"""Independent diagnostics for nuclear-norm proximal-gradient stationarity.

Objective F(X)=0.5||P_Omega(X-Y)||_F^2+lam||X||_*.
The proximal-gradient mapping G_t(X)=(X-prox_{t lam ||.||_*}(X-t grad f(X)))/t.
G_t(X)=0 iff X is a global minimizer of this convex objective (t>0).
"""
import numpy as np
from scipy.linalg import svd
from matrix_completion import _validate

def nuclear_objective(x,y,indices,lam):
    y,indices=_validate(y,indices)
    x=np.asarray(x,dtype=float)
    if x.shape!=y.shape or not np.all(np.isfinite(x)) or not np.isfinite(lam) or lam<0:
        raise ValueError('invalid objective inputs')
    residual=x[indices]-y[indices]
    return float(0.5*np.dot(residual,residual)+lam*np.sum(svd(x,compute_uv=False)))

def proximal_gradient_mapping_norm(x,y,indices,lam,step=1.0):
    y,indices=_validate(y,indices)
    x=np.asarray(x,dtype=float)
    if x.shape!=y.shape or not np.all(np.isfinite(x)) or not np.isfinite(lam) or lam<0 or not np.isfinite(step) or step<=0:
        raise ValueError('invalid stationarity inputs')
    z=x.copy()
    z[indices]-=step*(x[indices]-y[indices])
    u,s,vt=svd(z,full_matrices=False)
    prox=(u*np.maximum(s-step*lam,0))@vt
    return float(np.linalg.norm(x-prox)/step)
