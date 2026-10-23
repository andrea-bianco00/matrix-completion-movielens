"""Training-only integration smoke test; NEVER loads validation/test rating values."""
import json, sys
from pathlib import Path
from runner_30 import run_child, HERE

def main(source, out):
    plan=json.loads((HERE/'benchmark_plan_28.json').read_text())
    results=[]
    for family, configurations in plan['candidates'].items():
        for params in configurations:
            result=run_child(family,params,Path(source),120)
            results.append(result)
            print(family,params,result['status'],result.get('training_rmse'),flush=True)
            Path(out).write_text(json.dumps({'data_access':'train_only','results':results},indent=2))
    if len(results)!=11: raise RuntimeError('incorrect number of candidates')
    return int(any(r['status']!='ok' for r in results))
if __name__=='__main__':
    sys.exit(main(sys.argv[1],sys.argv[2]))
