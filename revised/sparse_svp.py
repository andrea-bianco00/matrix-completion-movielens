"""Experimental matrix-free SVP for observed-entry squared loss.

Stores the current iterate as low-rank factors and the gradient as sparse COO.
Uses ARPACK truncated SVD; approximate and not a drop-in exact replacement.
"""
from dataclasses import dataclass
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import LinearOperator, svds, ArpackNoConvergence

@dataclass
class SparseSVPResult:
    left: np.ndarray
    singular_values: np.ndarray
    right_t: np.ndarray
    train_rmse: list[float]
    iterations: int
    stop_reason: str
    dense_fallbacks: int = 0

    def predict(self, rows, cols):
        rows = np.asarray(rows, dtype=np.int64)
        cols = np.asarray(cols, dtype=np.int64)
        if rows.shape != cols.shape:
            raise ValueError('row/column index shapes differ')
        if np.any(rows < 0) or np.any(rows >= self.left.shape[0]) or np.any(cols < 0) or np.any(cols >= self.right_t.shape[1]):
            raise ValueError('prediction indices out of bounds')
        return np.sum((self.left[rows] * self.singular_values) * self.right_t[:, cols].T, axis=1)

    def to_dense(self):
        return (self.left * self.singular_values) @ self.right_t


def sparse_svp(shape, indices, values, *, rank, step=1., max_iter=100, tol=1e-6,
               svd_tol=1e-10, svd_maxiter=None,
               on_arpack_failure="raise", max_dense_elements=100_000):
    """Approximate SVP via matrix-free top-k SVD; requires rank < min(shape).

    The data must contain unique observed positions. The step is on the gradient
    of 0.5 * sum_observed (X_ij - Y_ij)^2. Convergence tests here are iterate
    change diagnostics, not certificates of stationarity or global optimality.
    """
    if len(shape) != 2 or any(not isinstance(d, (int, np.integer)) or d < 2 for d in shape):
        raise ValueError('invalid shape')
    m, n = map(int, shape)
    if not isinstance(rank, (int, np.integer)) or not 1 <= rank < min(m, n):
        raise ValueError('rank must be between 1 and min(shape)-1')
    if not np.isfinite(step) or step <= 0 or not isinstance(max_iter, (int, np.integer)) or max_iter < 1 or not np.isfinite(tol) or tol < 0 or not np.isfinite(svd_tol) or svd_tol < 0:
        raise ValueError('invalid optimization parameters')
    if on_arpack_failure not in ("raise", "dense"):
        raise ValueError('on_arpack_failure must be raise or dense')
    if not isinstance(max_dense_elements, (int, np.integer)) or max_dense_elements < 1:
        raise ValueError('max_dense_elements must be a positive integer')
    if svd_maxiter is not None and (not isinstance(svd_maxiter, (int, np.integer)) or svd_maxiter < 1):
        raise ValueError('svd_maxiter must be a positive integer or None')
    if len(indices) != 2:
        raise ValueError('indices must have two arrays')
    rows, cols = [np.asarray(a) for a in indices]
    vals = np.asarray(values, dtype=float)
    if (rows.ndim != 1 or cols.ndim != 1 or vals.ndim != 1 or
        not (len(rows) == len(cols) == len(vals)) or len(vals) == 0 or
        not np.issubdtype(rows.dtype, np.integer) or not np.issubdtype(cols.dtype, np.integer) or
        not np.all(np.isfinite(vals)) or np.any(rows < 0) or np.any(rows >= m) or
        np.any(cols < 0) or np.any(cols >= n)):
        raise ValueError('invalid observed data')
    if len(set(zip(rows.tolist(), cols.tolist()))) != len(rows):
        raise ValueError('duplicate observations')
    left = np.zeros((m, rank)); s = np.zeros(rank); right_t = np.zeros((rank, n))
    errors = []
    dense_fallbacks = 0
    if not np.any(vals):
        return SparseSVPResult(left, s, right_t, [0.0], 1, 'zero observed target')
    for iteration in range(1, max_iter + 1):
        pred = np.sum((left[rows] * s) * right_t[:, cols].T, axis=1)
        correction = coo_matrix((step * (vals - pred), (rows, cols)), shape=(m, n)).tocsr()
        def matvec(v):
            return left @ (s * (right_t @ v)) + correction @ v
        def rmatvec(v):
            return right_t.T @ (s * (left.T @ v)) + correction.T @ v
        def matmat(v):
            return left @ (s[:, None] * (right_t @ v)) + correction @ v
        def rmatmat(v):
            return right_t.T @ (s[:, None] * (left.T @ v)) + correction.T @ v
        operator = LinearOperator((m, n), matvec=matvec, rmatvec=rmatvec,
                                  matmat=matmat, rmatmat=rmatmat, dtype=float)
        try:
            u, new_s, vt = svds(operator, k=rank, which='LM', tol=svd_tol,
                                maxiter=svd_maxiter, v0=np.ones(min(m, n)))
        except ArpackNoConvergence:
            if on_arpack_failure != "dense" or m * n > max_dense_elements:
                raise
            # Exact SVD of the SAME pre-projection operator; never substitute
            # a different optimization update or silently accept partial vectors.
            dense_operator = operator.matmat(np.eye(n))
            full_u, full_s, full_vt = np.linalg.svd(dense_operator, full_matrices=False)
            u, new_s, vt = full_u[:, :rank], full_s[:rank], full_vt[:rank, :]
            dense_fallbacks += 1
        if not (np.all(np.isfinite(u)) and np.all(np.isfinite(new_s)) and np.all(np.isfinite(vt))):
            raise FloatingPointError('SVD returned non-finite factors')
        order = np.argsort(new_s)[::-1]
        u, new_s, vt = u[:, order], new_s[order], vt[order]
        # Relative iterate change, computed from low-rank factors, no dense array.
        old_norm2 = np.sum(s ** 2)
        new_norm2 = np.sum(new_s ** 2)
        cross = np.sum((left.T @ u) * (s[:, None] * new_s[None, :]) * (right_t @ vt.T))
        delta = np.sqrt(max(0., old_norm2 + new_norm2 - 2 * cross)) / max(1., np.sqrt(old_norm2))
        left, s, right_t = u, new_s, vt
        pred = np.sum((left[rows] * s) * right_t[:, cols].T, axis=1)
        errors.append(float(np.sqrt(np.mean((pred - vals) ** 2))))
        if delta <= tol:
            return SparseSVPResult(left, s, right_t, errors, iteration, 'relative iterate change', dense_fallbacks)
    return SparseSVPResult(left, s, right_t, errors, max_iter, 'max_iter', dense_fallbacks)
