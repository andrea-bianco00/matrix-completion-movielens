"""Prospective fit/predict runner: validation labels loaded strictly after fit completes."""
import argparse, json, os, resource, subprocess, sys, time
from pathlib import Path
import numpy as np
from runner_30 import HERE, ROOT, training_data
from validation_32 import validation_data, score_validation


def fit_model(family, params, rows, cols, y, shape):
    """Training-only interface. Does not accept validation arrays or labels."""
    sys.path.insert(0,str(HERE));sys.path.insert(0,str(ROOT/'revised'))
    plan=json.loads((HERE/'benchmark_plan_28.json').read_text())
    if params not in plan['candidates'].get(family,[]):
        raise ValueError('candidate is not in the frozen grid')
    if family=='global_mean':
        mu=float(np.mean(y))
        return lambda r,c: np.full(len(r),mu),0,'closed_form'
    if family=='user_item_bias':
        from bias_baseline_29 import fit_bias,predict_bias
        m=fit_bias(rows,cols,y,shape,regularization=float(params['regularization']))
        return lambda r,c: predict_bias(m,r,c),m['iterations'],m['stop_reason']
    if family=='SVP':
        from sparse_svp import sparse_svp
        m=sparse_svp(shape,(rows,cols),y,rank=int(params['rank']),**plan['solver_settings']['SVP'])
        return m.predict,m.iterations,m.stop_reason
    if family=='SVT':
        from scalable_19 import svt_matrix_free
        m=svt_matrix_free(shape,(rows,cols),y,threshold=float(params['threshold']),**plan['solver_settings']['SVT'])
        return m.predict,len(m.history),m.stop_reason
    if family=='ADMiRA_lab_variant':
        from scalable_19 import admira_matrix_free
        m=admira_matrix_free(shape,(rows,cols),y,rank=int(params['rank']),**plan['solver_settings']['ADMiRA_lab_variant'])
        return m.predict,len(m.history),m.stop_reason
    raise ValueError('unknown model')


def worker(family,params,source,*,evaluate=False):
    start=time.perf_counter()
    rows,cols,y,shape=training_data(source)
    predict,iterations,stop=fit_model(family,params,rows,cols,y,shape)
    pred_train=np.asarray(predict(rows,cols),dtype=float)
    if pred_train.shape!=y.shape or not np.all(np.isfinite(pred_train)):
        raise ValueError('invalid training predictions')
    result={'family':family,'params':params,'status':'ok','training_n':len(y),
            'training_rmse':float(np.sqrt(np.mean((pred_train-y)**2))),
            'iterations':iterations,'stop_reason':stop,'shape':shape,
            'evaluated_split':'train_and_validation' if evaluate else 'train_only'}
    if evaluate:
        vr,vc,vy,warm,vshape=validation_data(source)  # validation labels first enter process AFTER fit
        if vshape!=shape: raise ValueError('validation shape mismatch')
        result.update(score_validation(predict(vr,vc),vy,warm))
    result['elapsed_seconds']=time.perf_counter()-start
    result['peak_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return result


def run_child(family,params,source,seconds=120,*,evaluate=False):
    cmd=[sys.executable,str(Path(__file__).resolve()),'--worker',family,json.dumps(params,sort_keys=True),str(source)]
    if evaluate: cmd.append('--validation')
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    start=time.perf_counter()
    try: p=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=seconds)
    except subprocess.TimeoutExpired:
        return {'family':family,'params':params,'status':'timeout','wall_seconds':time.perf_counter()-start}
    if p.returncode:
        return {'family':family,'params':params,'status':'error','returncode':p.returncode,
                'stderr_tail':p.stderr[-1500:],'wall_seconds':time.perf_counter()-start}
    try: result=json.loads(p.stdout)
    except json.JSONDecodeError:
        return {'family':family,'params':params,'status':'invalid_output','stdout_tail':p.stdout[-1000:]}
    result['wall_seconds']=time.perf_counter()-start
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--worker',action='store_true');ap.add_argument('family');ap.add_argument('params');ap.add_argument('source');ap.add_argument('--validation',action='store_true');ap.add_argument('--timeout',type=float,default=120)
    a=ap.parse_args();params=json.loads(a.params)
    if a.worker:
        resource.setrlimit(resource.RLIMIT_AS,(2_147_483_648,2_147_483_648))
        print(json.dumps(worker(a.family,params,Path(a.source),evaluate=a.validation)))
    else: print(json.dumps(run_child(a.family,params,Path(a.source),a.timeout,evaluate=a.validation),indent=2))
