"""Proximal-gradient stationarity diagnostic for matrix-free nuclear-norm fitting.

Computes ||X - prox_{t lambda ||.||_*}(X-t P_Omega(X-Y))||_F / t.
A zero residual is equivalent to first-order optimality for the convex objective.
A nonzero residual is not a certified objective gap. The partial SVD is
numerical, so computed residuals have numerical approximation error.
"""
import numpy as np
from scipy.sparse import coo_matrix
from scalable_19 import _inputs, _operator, _top_svd, LowRankFit


def matrix_free_pg_residual(fit, shape, indices, values, *, threshold,
                            step=1., initial_rank=8, svd_tol=1e-11):
    if not isinstance(fit, LowRankFit):
        raise TypeError('fit must be LowRankFit')
    rows, cols, vals = _inputs(shape, indices, values)
    if (not np.isfinite(threshold) or threshold <= 0 or
        not np.isfinite(step) or not 0 < step <= 1 or
        not isinstance(initial_rank, (int, np.integer)) or
        not np.isfinite(svd_tol) or svd_tol < 0):
        raise ValueError('invalid parameters')
    m, n = shape
    cap = min(shape) - 1
    if not 1 <= initial_rank <= cap:
        raise ValueError('invalid initial rank')
    u, s, vt = fit.left, fit.singular_values, fit.right_t
    if (u.shape != (m, len(s)) or vt.shape != (len(s), n) or
        not np.all(np.isfinite(u)) or not np.all(np.isfinite(s)) or
        not np.all(np.isfinite(vt))):
        raise ValueError('invalid fit factors')
    if len(s) == 0 and not np.any(vals):
        return {'pg_mapping_norm': 0.0, 'pg_mapping_norm_relative': 0.0,
                'next_prox_rank': 0, 'computed_singular_values': 0}
    pred = np.sum((u[rows] * s) * vt[:, cols].T, axis=1)
    correction = coo_matrix((step * (vals - pred), (rows, cols)), shape=shape).tocsr()
    op = _operator(shape, u, s, vt, correction)
    k = initial_rank
    while True:
        nu, singular, nvt = _top_svd(op, k, svd_tol=svd_tol)
        if singular[-1] < step * threshold:
            break
        if k == cap:
            raise RuntimeError('cannot certify threshold cutoff using partial SVD')
        k = min(cap, max(k + 1, 2 * k))
    shrunk = np.maximum(singular - step * threshold, 0.)
    keep = shrunk > 0
    nu, ns, nvt = nu[:, keep], shrunk[keep], nvt[keep]
    # ||X-Z||_F^2 = ||X||_F^2 + ||Z||_F^2 - 2 <X,Z>.
    inner = np.sum((u.T @ nu) * (s[:, None] * ns[None, :]) * (vt @ nvt.T))
    squared = max(0., float(np.dot(s,s) + np.dot(ns,ns) - 2*inner))
    return {'pg_mapping_norm': float(np.sqrt(squared) / step),
            'pg_mapping_norm_relative': float(np.sqrt(squared) / step / max(1., np.linalg.norm(s))),
            'next_prox_rank': int(len(ns)), 'computed_singular_values': int(k)}
