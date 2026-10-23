"""Freeze selected validation winners without reading test data or labels."""
import hashlib, json
from pathlib import Path
from validation_32 import select_by_family
HERE=Path(__file__).resolve().parent

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def freeze_selection(plan_path, result_path, frozen_path, output_path):
    plan=json.loads(Path(plan_path).read_text())
    result=json.loads(Path(result_path).read_text())
    frozen=json.loads(Path(frozen_path).read_text())
    if sha(plan_path)!=frozen['sha256']['plan']:
        raise ValueError('plan differs from pre-validation freeze')
    if result['freeze_sha256']!=sha(frozen_path):
        raise ValueError('validation result references a different freeze')
    records=result['records']
    if len(records)!=sum(map(len,plan['candidates'].values())): raise ValueError('wrong number of candidates')
    selected=select_by_family(records,plan)
    if any(v is None for v in selected.values()): raise ValueError('no eligible candidate for a family')
    summary={family:{'params':entry['params'],'validation_rmse':entry['validation_rmse'],
                     'stop_reason':entry['stop_reason']} for family,entry in selected.items()}
    global_best=min(selected.items(),key=lambda item:item[1]['validation_rmse'])
    artifact={'status':'selection_frozen_before_test_evaluation','evaluation_policy':'no test labels accessed; do not modify candidates based on test',
              'source_hashes':{'plan':sha(plan_path),'validation_results':sha(result_path),'prior_freeze':sha(frozen_path)},
              'family_winners':summary,'global_winner':{'family':global_best[0],**summary[global_best[0]]}}
    Path(output_path).write_text(json.dumps(artifact,indent=2,sort_keys=True)+'\n')
    return artifact
if __name__=='__main__':
    freeze_selection(HERE/'benchmark_plan_28.json',HERE/'validation_results_34.json',HERE/'freeze_34.json',HERE/'selection_lock_41.json')
    print('Selection lock written without accessing test labels.')
