"""Reproduce the frozen global winner's TRAINING metric, without test scoring.

Requires a separately obtained source assignment archive matching the frozen
MovieLens ratings.csv checksum. Creates disjoint exports in a temporary folder;
the fit subprocess receives only fit/ and unlabeled training coordinates.
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'prospective'))
from isolation_36 import export, load_fit
from isolated_runner_37 import run_child
from final_dryrun_42 import verify_lock


def reproduce(source, *, seconds=120):
    lock = verify_lock()
    winner = lock['global_winner']
    if winner['family'] != 'user_item_bias' or winner['params'] != {'regularization': 5.0}:
        raise ValueError('unexpected frozen winner; refusing to proceed')
    with tempfile.TemporaryDirectory(prefix='mc_training_reproduction_') as temp:
        root = Path(temp)
        export(source, root / 'split')
        fit = root / 'split' / 'fit'
        rows, cols, y, shape = load_fit(fit)
        query = root / 'train_coordinates.npz'
        np.savez(query, rows=rows[:100], cols=cols[:100])
        run = run_child(winner['family'], winner['params'], fit, query, seconds=seconds)
        if run.get('status') != 'ok':
            raise RuntimeError(f'training failed: {run.get("status")}')
        expected = 0.8166725203779724  # Historical, already recorded in final_winner_results_43.json
        observed = run['training_rmse']
        if abs(observed - expected) > 1e-10:
            raise AssertionError(f'training RMSE differs: {observed} vs {expected}')
        return {'status': 'training_reproduced', 'training_n': len(y),
                'matrix_shape': list(shape), 'training_rmse': observed,
                'historical_training_rmse': expected, 'absolute_difference': abs(observed-expected),
                'family': winner['family'], 'params': winner['params'],
                'test_labels_scored': False,
                'caveat': 'The exporter reads and partitions the original archive; the fit worker receives only fit data and unlabeled coordinates. No OS sandbox is claimed.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path, help='Original archive containing ml-latest-small/ratings.csv')
    parser.add_argument('--seconds', type=int, default=120)
    args = parser.parse_args()
    print(json.dumps(reproduce(args.source, seconds=args.seconds), indent=2))
