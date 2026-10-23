"""Offline consistency checks for the published research artifacts; no data download or fitting."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
P = ROOT / 'prospective'


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root=ROOT):
    p = Path(root) / 'prospective'
    lock = json.loads((p / 'selection_lock_41.json').read_text())
    final = json.loads((p / 'final_winner_results_43.json').read_text())
    validation = json.loads((p / 'validation_results_34.json').read_text())
    manifest = json.loads((p / 'split_manifest_18.json').read_text())
    assert sum(manifest['counts'].values()) == manifest['source_rows'] == 100836
    for key, filename in [('plan', 'benchmark_plan_28.json'),
                          ('prior_freeze', 'freeze_34.json'),
                          ('validation_results', 'validation_results_34.json')]:
        assert sha256(p / filename) == lock['source_hashes'][key], f'hash mismatch: {filename}'
    assert sha256(p / 'selection_lock_41.json') == final['selection_lock_sha256']
    winners = lock['family_winners']
    best = min(winners, key=lambda k: winners[k]['validation_rmse'])
    assert best == lock['global_winner']['family'] == final['global_winner']
    assert winners[best]['params'] == lock['global_winner']['params']
    assert len(final['results']) == 1, 'only global winner should have test results'
    r = final['results'][0]
    assert r['family'] == best and r['params'] == lock['global_winner']['params']
    assert r['test_n'] == manifest['counts']['test']
    assert r['warm_item_n'] + r['cold_item_n'] == r['test_n']
    reconstructed = math.sqrt((r['warm_item_n'] * r['warm_item_rmse'] ** 2 +
                               r['cold_item_n'] * r['cold_item_rmse'] ** 2) / r['test_n'])
    assert math.isclose(reconstructed, r['test_rmse'], rel_tol=1e-12, abs_tol=1e-12)
    assert final['selected_before_test'] is True
    assert final['os_sandbox_verified'] is False
    assert validation, 'validation artifact must not be empty'
    return {'verified': True, 'family_winners': len(winners), 'test_n': r['test_n'],
            'test_rmse': r['test_rmse'], 'model': best, 'os_sandbox_verified': False}


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2, sort_keys=True))
