"""First-order tangent-gradient norm at a numerical rank-r matrix, sparse observations.

Necessary stationarity condition for smooth exactly-rank-r manifold only.
"""
import numpy as np

def tangent_gradient_sparse(fit, rows, cols, values, shape, rank, *, rank_tol=1e-10):
    rows=np.asarray(rows); cols=np.asarray(cols); values=np.asarray(values,dtype=float)
    u=np.asarray(fit.left); s=np.asarray(fit.singular_values); vt=np.asarray(fit.right_t)
    m,n=shape
    if (rows.ndim!=1 or cols.ndim!=1 or values.ndim!=1 or not len(values) or
        not(len(rows)==len(cols)==len(values)) or not np.issubdtype(rows.dtype,np.integer) or
        not np.issubdtype(cols.dtype,np.integer) or np.any(rows<0) or np.any(rows>=m) or
        np.any(cols<0) or np.any(cols>=n) or not np.all(np.isfinite(values)) or
        len(set(zip(rows.tolist(),cols.tolist())))!=len(rows)):
        raise ValueError('invalid observations')
    if not isinstance(rank,(int,np.integer)) or rank<1 or rank>min(m,n):
        raise ValueError('invalid rank')
    if u.shape!=(m,len(s)) or vt.shape!=(len(s),n) or not np.all(np.isfinite(u)) or not np.all(np.isfinite(s)) or not np.all(np.isfinite(vt)):
        raise ValueError('invalid factors')
    if len(s)!=rank or s[-1] <= rank_tol*max(1.,s[0]):
        raise ValueError('not an exact numerically stable rank-r point')
    # Factor columns must be orthonormal for the tangent projection identity.
    if not np.allclose(u.T@u,np.eye(rank),atol=1e-8) or not np.allclose(vt@vt.T,np.eye(rank),atol=1e-8):
        raise ValueError('nonorthonormal factors')
    residual=np.sum((u[rows]*s)*vt[:,cols].T,axis=1)-values
    # U^T G is r x n, and G V is m x r. Neither uses an m x n array.
    utg=np.zeros((rank,n)); gv=np.zeros((m,rank))
    np.add.at(utg,(slice(None),cols), (u[rows]*residual[:,None]).T)
    np.add.at(gv,rows,residual[:,None]*vt[:,cols].T)
    middle=utg@vt.T
    squared=np.sum(utg**2)+np.sum(gv**2)-np.sum(middle**2)
    return {'tangent_gradient_norm':float(np.sqrt(max(0.,squared))),
            'gradient_norm':float(np.linalg.norm(residual)),
            'observed_objective':float(0.5*np.dot(residual,residual)),
            'relative_tangent_gradient':float(np.sqrt(max(0.,squared))/max(1.,np.linalg.norm(residual)))}
