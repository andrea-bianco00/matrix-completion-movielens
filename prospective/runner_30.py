"""Pre-validation train-only runner. Never accesses validation/test rating values."""
import argparse, csv, hashlib, io, json, os, subprocess, sys, tempfile, time, zipfile, resource
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent

def training_data(source):
    manifest=json.loads((HERE/'split_manifest_18.json').read_text())
    with zipfile.ZipFile(source) as z:
        names=[n for n in z.namelist() if n.endswith('/ratings.csv')]
        if len(names)!=1: raise ValueError('ambiguous ratings source')
        raw=z.read(names[0])
    if hashlib.sha256(raw).hexdigest()!=manifest['source_sha256']: raise ValueError('source checksum mismatch')
    records=list(csv.DictReader(io.StringIO(raw.decode('utf-8'))))
    with np.load(HERE/'split_indices_18.npz') as p:
        train=p['train'].copy()
    if hashlib.sha256(train.astype('<i8').tobytes()).hexdigest()!=manifest['index_sha256']['train']:
        raise ValueError('train index checksum mismatch')
    if np.any(train<0) or np.any(train>=len(records)) or len(np.unique(train))!=len(train):
        raise ValueError('invalid training indices')
    # Build ID vocabulary without reading held-out rating values.
    users=sorted({int(r['userId']) for r in records})
    items=sorted({int(r['movieId']) for r in records})
    um={u:i for i,u in enumerate(users)}; im={v:i for i,v in enumerate(items)}
    selected=(records[int(i)] for i in train)
    rows=[]; cols=[]; values=[]
    for r in selected:
        rows.append(um[int(r['userId'])]); cols.append(im[int(r['movieId'])]); values.append(float(r['rating']))
    return np.asarray(rows,dtype=np.int64),np.asarray(cols,dtype=np.int64),np.asarray(values), (len(users),len(items))

def fit_train_only(family, params, source):
    rows,cols,y,shape=training_data(source)
    sys.path.insert(0,str(HERE));sys.path.insert(0,str(ROOT/'revised'))
    start=time.perf_counter()
    if family=='global_mean':
        pred=np.full(len(y),float(y.mean())); iterations=0; stop='closed_form'
    elif family=='user_item_bias':
        from bias_baseline_29 import fit_bias,predict_bias
        model=fit_bias(rows,cols,y,shape,regularization=float(params['regularization']))
        pred=predict_bias(model,rows,cols);iterations=model['iterations'];stop=model['stop_reason']
    elif family=='SVP':
        from sparse_svp import sparse_svp
        cfg=json.loads((HERE/'benchmark_plan_28.json').read_text())['solver_settings']['SVP']
        model=sparse_svp(shape,(rows,cols),y,rank=int(params['rank']),**cfg)
        pred=model.predict(rows,cols);iterations=model.iterations;stop=model.stop_reason
    elif family=='SVT':
        from scalable_19 import svt_matrix_free
        cfg=json.loads((HERE/'benchmark_plan_28.json').read_text())['solver_settings']['SVT']
        model=svt_matrix_free(shape,(rows,cols),y,threshold=float(params['threshold']),**cfg)
        pred=model.predict(rows,cols);iterations=len(model.history);stop=model.stop_reason
    elif family=='ADMiRA_lab_variant':
        from scalable_19 import admira_matrix_free
        cfg=json.loads((HERE/'benchmark_plan_28.json').read_text())['solver_settings']['ADMiRA_lab_variant']
        model=admira_matrix_free(shape,(rows,cols),y,rank=int(params['rank']),**cfg)
        pred=model.predict(rows,cols);iterations=len(model.history);stop=model.stop_reason
    else: raise ValueError('unsupported family')
    return {'family':family,'params':params,'status':'ok','training_rmse':float(np.sqrt(np.mean((pred-y)**2))),
            'iterations':iterations,'stop_reason':stop,'elapsed_seconds':time.perf_counter()-start,
            'training_n':len(y),'shape':shape,'evaluated_split':'train_only',
            'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}

def run_child(family,params,source,seconds):
    cmd=[sys.executable,str(Path(__file__).resolve()),'--worker',family,json.dumps(params,sort_keys=True),str(source)]
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    start=time.perf_counter()
    try:
        p=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=seconds)
    except subprocess.TimeoutExpired:
        return {'family':family,'params':params,'status':'timeout','wall_seconds':time.perf_counter()-start}
    if p.returncode:
        return {'family':family,'params':params,'status':'error','returncode':p.returncode,'stderr_tail':p.stderr[-1500:],'wall_seconds':time.perf_counter()-start}
    try: result=json.loads(p.stdout)
    except json.JSONDecodeError:
        return {'family':family,'params':params,'status':'invalid_output','stdout_tail':p.stdout[-1000:]}
    result['wall_seconds']=time.perf_counter()-start
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('family');parser.add_argument('params');parser.add_argument('source')
    parser.add_argument('--timeout',type=float,default=120)
    args=parser.parse_args()
    params=json.loads(args.params)
    if args.worker:
        # Hard per-worker address-space ceiling (Linux); parent process remains unrestricted.
        resource.setrlimit(resource.RLIMIT_AS,(2_147_483_648,2_147_483_648))
        print(json.dumps(fit_train_only(args.family,params,Path(args.source))))
    else:
        print(json.dumps(run_child(args.family,params,Path(args.source),args.timeout),indent=2))
