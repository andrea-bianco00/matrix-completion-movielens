"""Audited educational matrix completion implementations.

Training functions deliberately accept no validation or test data.
All arrays are dense; intended for small/moderate experiments.
"""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import svd

@dataclass
class FitResult:
    matrix: np.ndarray
    train_errors: list[float]  # observed-entry RMSE, not relative error
    iterations: int
    stop_reason: str
    diagnostics: list[dict] | None = None

def _validate(y, indices):
    y = np.asarray(y, dtype=float)
    if y.ndim != 2 or not np.all(np.isfinite(y)):
        raise ValueError('y must be a finite 2D array')
    if len(indices) != 2:
        raise ValueError('indices must be (rows, columns)')
    rows, cols = (np.asarray(v) for v in indices)
    if rows.ndim != 1 or cols.ndim != 1 or len(rows) != len(cols) or len(rows) == 0:
        raise ValueError('indices must be nonempty equal-length 1D arrays')
    if not np.issubdtype(rows.dtype, np.integer) or not np.issubdtype(cols.dtype, np.integer):
        raise ValueError('indices must be integer arrays')
    if np.any(rows < 0) or np.any(cols < 0) or np.any(rows >= y.shape[0]) or np.any(cols >= y.shape[1]):
        raise ValueError('indices out of bounds')
    if len(set(zip(rows.tolist(), cols.tolist()))) != len(rows):
        raise ValueError('duplicate observed indices')
    return y, (rows, cols)

def relative_error(x, truth, indices):
    """Relative L2 error over indexed entries; undefined for zero truth norm."""
    x = np.asarray(x, dtype=float)
    truth, indices = _validate(truth, indices)
    if x.shape != truth.shape:
        raise ValueError('shape mismatch')
    if not np.all(np.isfinite(x)):
        raise ValueError('prediction must be finite')
    denominator = np.linalg.norm(truth[indices])
    if denominator == 0:
        raise ValueError('relative error undefined: observed truth norm is zero')
    return float(np.linalg.norm(x[indices] - truth[indices]) / denominator)

def observed_rmse(x, y, indices):
    """Finite observed-entry RMSE; defined even when all targets are zero."""
    y, indices = _validate(y, indices)
    x = np.asarray(x, dtype=float)
    if x.shape != y.shape or not np.all(np.isfinite(x)):
        raise ValueError("prediction shape or values invalid")
    return float(np.sqrt(np.mean((x[indices] - y[indices]) ** 2)))

def truncated_svd(a, rank):
    """Deterministic dense truncated SVD (reference, not scalable implementation)."""
    a = np.asarray(a, dtype=float)
    if not isinstance(rank, (int, np.integer)) or not 1 <= rank <= min(a.shape):
        raise ValueError('invalid rank')
    u, s, vt = svd(a, full_matrices=False)
    return u[:, :rank], s[:rank], vt[:rank]

def _project_rank(a, rank):
    u, s, vt = truncated_svd(a, rank)
    return (u * s) @ vt

def _finish(x, history, reason):
    return FitResult(x, history, len(history), reason)

def svp(y, indices, *, rank, step=1.0, max_iter=100, tol=1e-6):
    """Projected gradient on observed squared loss with rank constraint."""
    y, indices = _validate(y, indices)
    if not isinstance(rank, (int, np.integer)) or not 1 <= rank <= min(y.shape) or step <= 0 or not np.isfinite(step) or max_iter < 1 or tol < 0:
        raise ValueError('invalid optimization parameter')
    x = np.zeros_like(y)
    history = []
    for _ in range(max_iter):
        z = x.copy()
        z[indices] += step * (y[indices] - x[indices])
        next_x = _project_rank(z, rank)
        err = observed_rmse(next_x, y, indices)
        history.append(err)
        delta = np.linalg.norm(next_x - x) / max(1.0, np.linalg.norm(x))
        x = next_x
        if delta <= tol:
            return _finish(x, history, 'relative iterate change')
    return _finish(x, history, 'max_iter')

def svt_proximal(y, indices, *, threshold, step=1.0, max_iter=100, tol=1e-6):
    """Proximal-gradient nuclear-norm regularization.

    Minimizes 0.5*||P_Omega(X-Y)||_F^2 + threshold*||X||_* when
    step=1. Thresholding at each iteration is step*threshold.
    """
    y, indices = _validate(y, indices)
    if threshold < 0 or not np.isfinite(threshold) or not (0 < step <= 1) or max_iter < 1 or tol < 0:
        raise ValueError('invalid optimization parameter')
    x = np.zeros_like(y)
    history = []
    for _ in range(max_iter):
        z = x.copy()
        z[indices] += step * (y[indices] - x[indices])
        u, s, vt = svd(z, full_matrices=False)
        s = np.maximum(s - step * threshold, 0)
        next_x = (u * s) @ vt
        err = observed_rmse(next_x, y, indices)
        history.append(err)
        delta = np.linalg.norm(next_x - x) / max(1.0, np.linalg.norm(x))
        x = next_x
        if delta <= tol:
            return _finish(x, history, 'relative iterate change')
    return _finish(x, history, 'max_iter')

def admira_lab(y, indices, *, rank, max_iter=100, tol=1e-6, track_conditioning=False, lstsq_rcond=None):
    """ADMiRA-style algorithm matching lab steps (no recovery guarantee claimed)."""
    y, indices = _validate(y, indices)
    if not isinstance(rank, (int, np.integer)) or not 1 <= rank <= min(y.shape) or max_iter < 1 or tol < 0:
        raise ValueError('invalid optimization parameter')
    if lstsq_rcond is not None and (not np.isfinite(lstsq_rcond) or lstsq_rcond < 0):
        raise ValueError('invalid lstsq_rcond')
    x = np.zeros_like(y)
    atoms = []
    diagnostics = [] if track_conditioning else None
    history = []
    for _ in range(max_iter):
        residual = np.zeros_like(y)
        residual[indices] = y[indices] - x[indices]
        u, _, vt = truncated_svd(residual, min(2 * rank, min(y.shape)))
        candidates = atoms + [np.outer(u[:, j], vt[j]) for j in range(u.shape[1])]
        design = np.column_stack([a[indices] for a in candidates])
        coefficients, *_ = np.linalg.lstsq(design, y[indices], rcond=lstsq_rcond)
        if track_conditioning:
            from conditioning import design_diagnostics
            diagnostic = design_diagnostics(design, rcond=lstsq_rcond)
            diagnostic['iteration'] = len(history) + 1
            diagnostic['least_squares_residual_norm'] = float(np.linalg.norm(design @ coefficients - y[indices]))
            diagnostic['coefficient_norm'] = float(np.linalg.norm(coefficients))
            diagnostics.append(diagnostic)
        intermediate = sum((c * a for c, a in zip(coefficients, candidates)), np.zeros_like(y))
        u2, s2, vt2 = truncated_svd(intermediate, rank)
        next_x = (u2 * s2) @ vt2
        atoms = [np.outer(u2[:, j], vt2[j]) for j in range(rank)]
        history.append(observed_rmse(next_x, y, indices))
        delta = np.linalg.norm(next_x - x) / max(1.0, np.linalg.norm(x))
        x = next_x
        if delta <= tol:
            return FitResult(x, history, len(history), 'relative iterate change', diagnostics)
    return FitResult(x, history, len(history), 'max_iter', diagnostics)

def evaluate(x, y, indices):
    """RMSE and MAE on a held-out index set; call only after model selection."""
    y, indices = _validate(y, indices)
    if x.shape != y.shape:
        raise ValueError('shape mismatch')
    if not np.all(np.isfinite(x)):
        raise ValueError('prediction must be finite')
    errors = x[indices] - y[indices]
    return {'rmse': float(np.sqrt(np.mean(errors**2))), 'mae': float(np.mean(np.abs(errors)))}
