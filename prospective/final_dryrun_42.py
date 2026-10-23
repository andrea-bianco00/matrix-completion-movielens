"""Synthetic-only preflight for sealed final evaluation; never reads MovieLens test labels."""
import hashlib
import json
from pathlib import Path
import numpy as np
from final_preflight_41 import HERE, sha
from validation_32 import score_validation

LOCK=HERE/'selection_lock_41.json'

def verify_lock(lock_path=LOCK):
    lock=json.loads(Path(lock_path).read_text())
    expected={'plan':HERE/'benchmark_plan_28.json',
              'validation_results':HERE/'validation_results_34.json',
              'prior_freeze':HERE/'freeze_34.json'}
    if lock.get('status')!='selection_frozen_before_test_evaluation':
        raise ValueError('selection is not frozen')
    for name,path in expected.items():
        if lock['source_hashes'].get(name)!=sha(path):
            raise ValueError(f'{name} changed since selection lock')
    winner=lock['global_winner']
    if winner['family'] not in lock['family_winners'] or winner!=dict(family=winner['family'],**lock['family_winners'][winner['family']]):
        raise ValueError('inconsistent global winner')
    return lock

def score_sealed_predictions(predictions, labels, warm):
    """Scorer takes arrays, not filesystem paths; validates all inputs."""
    pred=np.asarray(predictions,dtype=float)
    truth=np.asarray(labels,dtype=float)
    warm=np.asarray(warm)
    if warm.dtype.kind!='b': raise ValueError('warm mask must be boolean')
    if truth.ndim!=1 or pred.ndim!=1 or warm.ndim!=1 or len(truth)==0:
        raise ValueError('expected nonempty one-dimensional arrays')
    if pred.shape!=truth.shape or warm.shape!=truth.shape:
        raise ValueError('array shape mismatch')
    if not np.all(np.isfinite(pred)) or not np.all(np.isfinite(truth)):
        raise ValueError('nonfinite values')
    return score_validation(pred,truth,warm)

def synthetic_dryrun():
    lock=verify_lock()
    # Entirely fabricated values, independent of MovieLens test and validation labels.
    predictions=np.array([3.,4.,2.,1.])
    synthetic_truth=np.array([4.,4.,1.,2.])
    warm=np.array([True,True,False,False])
    result=score_sealed_predictions(predictions,synthetic_truth,warm)
    expected=np.sqrt(3/4)
    if not np.isclose(result['validation_rmse'],expected,rtol=0,atol=1e-14):
        raise AssertionError('synthetic RMSE mismatch')
    return {'status':'synthetic_only_preflight_passed',
            'test_labels_accessed':False,
            'selection_family':lock['global_winner']['family'],
            'synthetic_n':len(predictions),'synthetic_rmse':result['validation_rmse'],
            'synthetic_warm_rmse':result['warm_item_rmse'],
            'synthetic_cold_rmse':result['cold_item_rmse']}

if __name__=='__main__':
    print(json.dumps(synthetic_dryrun(),indent=2))
