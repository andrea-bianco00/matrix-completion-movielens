"""Fit using fit/ only; predict on unlabeled coordinates. Scoring is outside worker.
This is API/data minimization, NOT an OS sandbox.
"""
import argparse, json, os, resource, subprocess, sys, time
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
from isolation_36 import load_fit
from runner_33 import fit_model

def fit_predict(family, params, fit_dir, coordinate_file):
    rows,cols,y,shape=load_fit(fit_dir)
    with np.load(coordinate_file,allow_pickle=False) as z:
        qr=np.asarray(z['rows'],dtype=np.int64)
        qc=np.asarray(z['cols'],dtype=np.int64)
    if qr.ndim!=1 or qc.ndim!=1 or qr.shape!=qc.shape: raise ValueError('bad query shape')
    if np.any(qr<0) or np.any(qr>=shape[0]) or np.any(qc<0) or np.any(qc>=shape[1]):
        raise ValueError('query indices out of bounds')
    predict,iters,stop=fit_model(family,params,rows,cols,y,shape)
    pred=np.asarray(predict(qr,qc),dtype=float)
    if pred.shape!=qr.shape or not np.all(np.isfinite(pred)): raise ValueError('invalid predictions')
    train_pred=np.asarray(predict(rows,cols),dtype=float)
    if train_pred.shape!=y.shape or not np.all(np.isfinite(train_pred)): raise ValueError('invalid train predictions')
    return {'family':family,'params':params,'status':'ok','training_n':len(y),
            'training_rmse':float(np.sqrt(np.mean((train_pred-y)**2))),
            'query_n':len(qr),'predictions':pred.tolist(),'iterations':iters,'stop_reason':stop,
            'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}

def run_child(family,params,fit_dir,coordinate_file,seconds=120):
    cmd=[sys.executable,str(Path(__file__).resolve()),'--worker',family,json.dumps(params,sort_keys=True),str(fit_dir),str(coordinate_file)]
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    t=time.perf_counter()
    try: p=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=seconds)
    except subprocess.TimeoutExpired: return {'status':'timeout','wall_seconds':time.perf_counter()-t}
    if p.returncode: return {'status':'error','returncode':p.returncode,'stderr_tail':p.stderr[-1000:]}
    try: result=json.loads(p.stdout)
    except json.JSONDecodeError: return {'status':'invalid_output'}
    result['wall_seconds']=time.perf_counter()-t
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');p.add_argument('family');p.add_argument('params');p.add_argument('fit_dir');p.add_argument('coordinate_file');a=p.parse_args()
    if a.worker:
        resource.setrlimit(resource.RLIMIT_AS,(2_147_483_648,2_147_483_648))
        print(json.dumps(fit_predict(a.family,json.loads(a.params),a.fit_dir,a.coordinate_file)))
    else:
        print(json.dumps(run_child(a.family,json.loads(a.params),a.fit_dir,a.coordinate_file),indent=2))
