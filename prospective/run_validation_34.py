"""Frozen 11-candidate validation benchmark; no test labels read."""
import hashlib,json,sys,time
from pathlib import Path
from runner_33 import run_child
from validation_32 import select_by_family
HERE=Path(__file__).resolve().parent
SOURCE=Path('/mnt/data/Dimension Reduction and Matrix Completion.zip')
PLAN=HERE/'benchmark_plan_28.json'
RUNNER=HERE/'runner_33.py'
VALID=HERE/'validation_32.py'
FREEZE=HERE/'freeze_34.json'
OUTPUT=HERE/'validation_results_34.json'
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    if not FREEZE.exists():
        raise RuntimeError('freeze manifest missing')
    freeze=json.loads(FREEZE.read_text())
    for name,path in [('plan',PLAN),('runner',RUNNER),('validation',VALID),('source',SOURCE)]:
        if sha(path)!=freeze['sha256'][name]: raise RuntimeError('checksum mismatch '+name)
    plan=json.loads(PLAN.read_text()); records=json.loads(OUTPUT.read_text())['records'] if OUTPUT.exists() else []
    for family,configs in plan['candidates'].items():
        for params in configs:
            if any(r['family']==family and r['params']==params for r in records):
                continue
            result=run_child(family,params,SOURCE,seconds=120,evaluate=True)
            records.append(result)
            OUTPUT.write_text(json.dumps({'freeze_sha256':sha(FREEZE),'records':records,'complete':False},indent=2))
            print(family,params,result.get('status'),result.get('validation_rmse'),flush=True)
    selected=select_by_family(records,plan)
    OUTPUT.write_text(json.dumps({'freeze_sha256':sha(FREEZE),'records':records,'complete':True,'selected':selected},indent=2))
if __name__=='__main__': main()
