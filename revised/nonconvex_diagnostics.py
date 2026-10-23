"""Independent first-order diagnostics for fixed-rank matrix completion.

These are necessary stationarity checks at smooth points of the rank-r manifold,
not certificates of local/global optimality or ADMiRA convergence.
"""
import numpy as np
from scipy.linalg import svd
from matrix_completion import _validate

def observed_objective(x, y, indices):
    y, indices = _validate(y, indices)
    x = np.asarray(x, dtype=float)
    if x.shape != y.shape or not np.all(np.isfinite(x)):
        raise ValueError('invalid prediction')
    d = x[indices] - y[indices]
    return float(0.5 * np.dot(d, d))

def tangent_gradient_norm(x, y, indices, rank, *, rank_tol=1e-10):
    """Norm of gradient projected to tangent space of smooth rank-r manifold.

    Valid only if X has numerical rank exactly r with a clearly positive r-th
    singular value. The manifold stationarity condition is necessary, not sufficient.
    """
    y, indices = _validate(y, indices)
    x = np.asarray(x, dtype=float)
    if x.shape != y.shape or not np.all(np.isfinite(x)):
        raise ValueError('invalid prediction')
    if not isinstance(rank, (int, np.integer)) or not 1 <= rank <= min(x.shape):
        raise ValueError('invalid rank')
    u, s, vt = svd(x, full_matrices=False)
    if s[rank - 1] <= rank_tol * max(1., s[0]):
        raise ValueError('X does not have a numerically stable rank-r manifold point')
    if rank < len(s) and s[rank] > rank_tol * max(1., s[0]):
        raise ValueError('X exceeds the requested rank')
    grad = np.zeros_like(x)
    grad[indices] = x[indices] - y[indices]
    ur = u[:, :rank]
    vr = vt[:rank, :].T
    # Projection onto tangent: UU^T G + G VV^T - UU^T G VV^T.
    left = ur @ (ur.T @ grad)
    right = (grad @ vr) @ vr.T
    tangent = left + right - (left @ vr) @ vr.T
    return float(np.linalg.norm(tangent))

def svp_projected_step_norm(x, y, indices, rank, step=1.):
    """Fixed-point residual for projected-gradient rank<=r iteration.

    Zero means fixed point of *this update*, not a global optimum.
    """
    y, indices = _validate(y, indices)
    x = np.asarray(x, dtype=float)
    if x.shape != y.shape or not np.all(np.isfinite(x)):
        raise ValueError('invalid prediction')
    if not isinstance(rank, (int, np.integer)) or not 1 <= rank <= min(x.shape):
        raise ValueError('invalid rank')
    if not np.isfinite(step) or step <= 0:
        raise ValueError('invalid step')
    z = x.copy()
    z[indices] += step * (y[indices] - x[indices])
    u, s, vt = svd(z, full_matrices=False)
    projected = (u[:, :rank] * s[:rank]) @ vt[:rank]
    return float(np.linalg.norm(projected - x) / step)
