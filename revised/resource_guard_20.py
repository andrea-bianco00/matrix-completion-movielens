"""Pre-allocation resource checks for experimental matrix-free solvers."""
import numpy as np


def design_memory_estimate(n_observations, current_atoms, rank):
    """Conservative lower bound for dense float64 LS design storage in bytes."""
    for x in (n_observations, current_atoms, rank):
        if not isinstance(x, (int, np.integer)) or x < 0:
            raise ValueError('counts must be nonnegative integers')
    return int(n_observations) * (int(current_atoms) + 2 * int(rank)) * 8


def enforce_design_budget(n_observations, current_atoms, rank, max_design_bytes):
    if not isinstance(max_design_bytes, (int, np.integer)) or max_design_bytes < 1:
        raise ValueError('max_design_bytes must be a positive integer')
    needed = design_memory_estimate(n_observations, current_atoms, rank)
    if needed > max_design_bytes:
        raise MemoryError(f'ADMiRA LS design requires at least {needed} bytes; budget={max_design_bytes}')
    return needed
