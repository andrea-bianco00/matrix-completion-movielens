"""Experimental matrix-free SVT and ADMiRA-style solvers.

SVT uses adaptive partial SVD and refuses to silently truncate singular
values above the threshold. ADMiRA follows the audited *lab variant*, not a
claim of exact equivalence with the published ADMiRA algorithm.
"""
from dataclasses import dataclass
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import LinearOperator, svds
from scipy.linalg import svd
from resource_guard_20 import enforce_design_budget

@dataclass
class LowRankFit:
    left: np.ndarray
    singular_values: np.ndarray
    right_t: np.ndarray
    history: list
    stop_reason: str

    def to_dense(self):
        return (self.left * self.singular_values) @ self.right_t

    def predict(self, rows, cols):
        rows = np.asarray(rows); cols = np.asarray(cols)
        return np.sum((self.left[rows] * self.singular_values) * self.right_t[:, cols].T, axis=1)


def _inputs(shape, indices, values):
    if len(shape) != 2 or min(shape) < 2:
        raise ValueError('invalid shape')
    rows, cols = map(np.asarray, indices)
    vals = np.asarray(values, float)
    if (rows.ndim != 1 or cols.ndim != 1 or vals.ndim != 1 or
        not len(vals) or not (len(rows) == len(cols) == len(vals)) or
        not np.issubdtype(rows.dtype, np.integer) or not np.issubdtype(cols.dtype, np.integer) or
        np.any(rows < 0) or np.any(cols < 0) or np.any(rows >= shape[0]) or
        np.any(cols >= shape[1]) or not np.all(np.isfinite(vals)) or
        len(set(zip(rows.tolist(), cols.tolist()))) != len(rows)):
        raise ValueError('invalid observations')
    return rows, cols, vals


def _operator(shape, u, s, vt, correction):
    def matmat(v):
        return u @ (s[:, None] * (vt @ v)) + correction @ v
    def rmatmat(v):
        return vt.T @ (s[:, None] * (u.T @ v)) + correction.T @ v
    return LinearOperator(shape, matvec=lambda v: matmat(v[:, None])[:, 0],
                          rmatvec=lambda v: rmatmat(v[:, None])[:, 0],
                          matmat=matmat, rmatmat=rmatmat, dtype=float)


def _top_svd(op, k, *, svd_tol=1e-11):
    u, s, vt = svds(op, k=k, tol=svd_tol, which='LM', v0=np.ones(min(op.shape)))
    order = np.argsort(s)[::-1]
    return u[:, order], s[order], vt[order]


def svt_matrix_free(shape, indices, values, *, threshold, step=1., max_iter=50,
                    tol=1e-6, initial_rank=8, svd_tol=1e-11, diagnostic_log=None):
    """Proximal gradient for observed squared loss + nuclear norm.

    Exact soft-thresholding requires all singular values above step*threshold.
    We adaptively grow k until the smallest computed singular value is below
    threshold. If unable to certify the cutoff with partial SVD, raise.
    """
    rows, cols, vals = _inputs(shape, indices, values)
    if threshold <= 0 or not np.isfinite(threshold) or not 0 < step <= 1 or max_iter < 1 or tol < 0:
        raise ValueError('invalid parameters')
    m, n = shape; cap = min(m, n) - 1
    if not 1 <= initial_rank <= cap:
        raise ValueError('initial_rank must be less than min(shape)')
    u = np.zeros((m, 0)); s = np.empty(0); vt = np.zeros((0, n)); history = []
    for _ in range(max_iter):
        pred = np.sum((u[rows] * s) * vt[:, cols].T, axis=1)
        correction = coo_matrix((step * (vals - pred), (rows, cols)), shape=shape).tocsr()
        op = _operator(shape, u, s, vt, correction)
        k = initial_rank
        while True:
            new_u, singular, new_vt = _top_svd(op, k, svd_tol=svd_tol)
            # Boundary equality is unsafe; ask for more singular vectors.
            if singular[-1] < step * threshold:
                break
            if k == cap:
                raise RuntimeError('cannot establish SVT threshold cutoff with partial SVD')
            k = min(cap, max(k + 1, 2 * k))
        shrunk = np.maximum(singular - step * threshold, 0.)
        keep = shrunk > 0
        new_u, new_s, new_vt = new_u[:, keep], shrunk[keep], new_vt[keep]
        # Frobenius difference via low-rank inner products.
        cross = np.sum((u.T @ new_u) * (s[:, None] * new_s[None, :]) * (vt @ new_vt.T))
        change = np.sqrt(max(0., np.sum(s*s) + np.sum(new_s*new_s) - 2*cross)) / max(1., np.linalg.norm(s))
        u, s, vt = new_u, new_s, new_vt
        p = np.sum((u[rows] * s) * vt[:, cols].T, axis=1)
        history.append(float(np.sqrt(np.mean((p - vals)**2))))
        if diagnostic_log is not None:
            # Objective: 0.5*||P_Omega(X)-y||_2^2 + lambda*||X||_*,
            # where s are singular values of the current iterate.
            objective = 0.5 * float(np.sum((p - vals)**2)) + threshold * float(np.sum(s))
            diagnostic_log.append({'iteration': len(history), 'relative_change': float(change),
                                   'objective': objective, 'nuclear_norm': float(np.sum(s)),
                                   'rank': int(len(s))})
        if change <= tol:
            return LowRankFit(u, s, vt, history, 'relative iterate change')
    return LowRankFit(u, s, vt, history, 'max_iter')


def admira_matrix_free(shape, indices, values, *, rank, max_iter=20, tol=1e-6,
                       svd_tol=1e-11, lstsq_rcond=None, max_design_bytes=256_000_000,
                       conditioning_log=None):
    """Matrix-free implementation of the audited educational ADMiRA-style variant.

    Candidate atoms are retained as rank-one factors, and least squares is
    solved on observed entries. The 2r residual singular vectors are obtained
    by partial SVD, so 2r must be strictly less than min(shape).
    """
    rows, cols, vals = _inputs(shape, indices, values)
    m, n = shape
    if not isinstance(rank, (int, np.integer)) or rank < 1 or 2*rank >= min(shape) or max_iter < 1 or tol < 0:
        raise ValueError('invalid parameters or rank too large for partial SVD')
    u = np.zeros((m, 0)); s = np.empty(0); vt = np.zeros((0, n)); history = []
    if not np.any(vals):
        return LowRankFit(u, s, vt, [0.0], 'zero observed target')
    for _ in range(max_iter):
        pred = np.sum((u[rows] * s) * vt[:, cols].T, axis=1)
        residual = coo_matrix((vals - pred, (rows, cols)), shape=shape).tocsr()
        residual_op = _operator(shape, np.zeros((m, 0)), np.empty(0), np.zeros((0, n)), residual)
        ru, _, rvt = _top_svd(residual_op, 2*rank, svd_tol=svd_tol)
        cand_u = np.column_stack([u, ru])
        cand_vt = np.vstack([vt, rvt])
        enforce_design_budget(len(vals), cand_u.shape[1] - 2*rank, rank, max_design_bytes)
        design = cand_u[rows] * cand_vt[:, cols].T
        coef, _, design_rank, design_singular = np.linalg.lstsq(design, vals, rcond=lstsq_rcond)
        if conditioning_log is not None:
            # Record the numerical rank returned by the same least-squares solve.
            # The 2-norm condition number is infinite for rank-deficient designs.
            smallest = float(design_singular[-1]) if len(design_singular) else 0.0
            condition = (float(design_singular[0] / smallest)
                         if design_rank == design.shape[1] and smallest > 0 else float('inf'))
            conditioning_log.append({'iteration': len(history) + 1,
                                     'columns': int(design.shape[1]),
                                     'numerical_rank': int(design_rank),
                                     'condition_2': condition,
                                     'singular_max': float(design_singular[0]),
                                     'singular_min': smallest})
        # Small core SVD is algebraically equivalent to SVD of sum of atoms.
        q_u, r_u = np.linalg.qr(cand_u, mode='reduced')
        q_v, r_v = np.linalg.qr(cand_vt.T, mode='reduced')
        core = (r_u * coef) @ r_v.T
        cu, cs, cvt = svd(core, full_matrices=False)
        new_u, new_s, new_vt = (q_u @ cu[:, :rank], cs[:rank], cvt[:rank] @ q_v.T)
        cross = np.sum((u.T @ new_u) * (s[:, None] * new_s[None, :]) * (vt @ new_vt.T))
        change = np.sqrt(max(0., np.sum(s*s) + np.sum(new_s*new_s) - 2*cross)) / max(1., np.linalg.norm(s))
        u, s, vt = new_u, new_s, new_vt
        p = np.sum((u[rows] * s) * vt[:, cols].T, axis=1)
        history.append(float(np.sqrt(np.mean((p - vals)**2))))
        if change <= tol:
            return LowRankFit(u, s, vt, history, 'relative iterate change')
    return LowRankFit(u, s, vt, history, 'max_iter')
