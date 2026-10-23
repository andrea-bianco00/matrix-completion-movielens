"""One-shot, preselected family-winner test evaluation. Not an OS sandbox.
No test labels are passed to model-fitting subprocesses; test scoring occurs after fits.
"""
import argparse,csv,hashlib,json,tempfile
from pathlib import Path
import numpy as np
from final_dryrun_42 import verify_lock,score_sealed_predictions
from isolation_36 import export,load_fit
from isolated_runner_37 import run_child
HERE=Path(__file__).resolve().parent
SOURCE=Path('/mnt/data/Dimension Reduction and Matrix Completion.zip')
OUT=HERE/'final_results_43.json'

def prepare_queries(fit_dir, sealed_csv, query_file):
    """Coordinator reads only test coordinates for model query preparation.
    Test labels remain in sealed_csv, inaccessible to fit API, but not OS isolated.
    """
    catalog=json.loads((fit_dir/'catalog.json').read_text())
    um={u:i for i,u in enumerate(catalog['users'])}
    im={v:i for i,v in enumerate(catalog['items'])}
    with sealed_csv.open(newline='') as f:
        coords=[(um[int(r['userId'])],im[int(r['movieId'])]) for r in csv.DictReader(f)]
    rows=np.array([r for r,c in coords],dtype=np.int64)
    cols=np.array([c for r,c in coords],dtype=np.int64)
    np.savez(query_file,rows=rows,cols=cols)
    return cols

def evaluate(source=SOURCE, output=OUT):
    lock=verify_lock()
    if output.exists(): raise FileExistsError('one-shot final result already exists; refusing repeat')
    with tempfile.TemporaryDirectory(prefix='mc_final_43_') as td:
        root=Path(td); export(source,root/'split')
        fit=root/'split'/'fit'; sealed=root/'split'/'sealed_test'/'test.csv'
        q=root/'query.npz'; cols=prepare_queries(fit,sealed,q)
        _,train_cols,_,_=load_fit(fit)
        warm=np.isin(cols,np.unique(train_cols))
        records=[]
        # Fit all previously selected family winners before opening test labels for scoring.
        for family,chosen in lock['family_winners'].items():
            run=run_child(family,chosen['params'],fit,q,seconds=120)
            if run['status']!='ok':
                raise RuntimeError(f'preselected fit failed: {family} {run["status"]}; no test scored')
            if len(run['predictions'])!=len(cols): raise ValueError('wrong prediction length')
            records.append({'family':family,'params':chosen['params'],'training_rmse':run['training_rmse'],
                            'iterations':run['iterations'],'stop_reason':run['stop_reason'],
                            'wall_seconds':run['wall_seconds'],'peak_rss_kib':run['peak_rss_kib'],
                            '_predictions':run['predictions']})
        with sealed.open(newline='') as f:
            labels=np.array([float(r['rating']) for r in csv.DictReader(f)],dtype=float)
        for rec in records:
            score=score_sealed_predictions(rec.pop('_predictions'),labels,warm)
            rec.update({('test_'+k[len('validation_'):] if k.startswith('validation_') else k):v for k,v in score.items()})
        result={'status':'one_shot_final_evaluation','selection_lock_sha256':hashlib.sha256((HERE/'selection_lock_41.json').read_bytes()).hexdigest(),
                'selected_before_test':True,'test_n':len(labels),'catalog_policy':'full user/movie IDs known in advance',
                'os_sandbox_verified':False,'results':records,
                'global_winner':lock['global_winner']['family'],
                'limitations':['Prior exploratory audits examined MovieLens under other splits',
                               'Test coordinates are available to prediction workers; test labels are not passed',
                               'OS-level Docker isolation has not been executed in this environment',
                               'No hyperparameter adjustments after test evaluation']}
        output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
        return result

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--execute',action='store_true');args=a.parse_args()
    if not args.execute: raise SystemExit('Explicit --execute required')
    print(json.dumps(evaluate(),indent=2))
