"""Diagnostics for least-squares atom design; no claims of identifiability."""
import numpy as np

def design_diagnostics(design, *, rcond=None):
    """Return SVD-based numerical rank, singular values and condition number.

    With rcond=None, use eps*max(m,n), matching numpy.linalg.lstsq default
    cutoff convention for numerical rank. Rank/condition refer to design,
    not to the unknown completed matrix.
    """
    a = np.asarray(design, dtype=float)
    if a.ndim != 2 or not np.isfinite(a).all():
        raise ValueError('design must be a finite 2D matrix')
    if rcond is not None and (not np.isfinite(rcond) or rcond < 0):
        raise ValueError('invalid rcond')
    singular = np.linalg.svd(a, compute_uv=False)
    cutoff_factor = np.finfo(float).eps * max(a.shape) if rcond is None else rcond
    cutoff = cutoff_factor * singular[0] if singular.size else 0.0
    rank = int(np.sum(singular > cutoff))
    condition = float(singular[0] / singular[-1]) if singular.size and singular[-1] > 0 else float('inf')
    return {'rank': rank, 'columns': a.shape[1], 'singular_values': singular.tolist(),
            'cutoff': float(cutoff), 'condition_number': condition,
            'rank_deficient': rank < a.shape[1]}
