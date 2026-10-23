"""Prospective evaluation utilities. No test split API or test rating reads."""
import csv, hashlib, io, json, zipfile
from pathlib import Path
import numpy as np
from runner_30 import HERE, training_data


def validation_data(source):
    """Return validation labels only to the scoring layer, never the fit layer."""
    manifest=json.loads((HERE/'split_manifest_18.json').read_text())
    with zipfile.ZipFile(source) as z:
        names=[n for n in z.namelist() if n.endswith('/ratings.csv')]
        if len(names)!=1: raise ValueError('ambiguous ratings source')
        raw=z.read(names[0])
    if hashlib.sha256(raw).hexdigest()!=manifest['source_sha256']:
        raise ValueError('source checksum mismatch')
    records=list(csv.DictReader(io.StringIO(raw.decode('utf-8'))))
    if len(records)!=manifest['source_rows']: raise ValueError('row count mismatch')
    with np.load(HERE/'split_indices_18.npz') as p:
        train=p['train'].copy(); val=p['validation'].copy()
    for name, indices in [('train',train),('validation',val)]:
        if hashlib.sha256(indices.astype('<i8').tobytes()).hexdigest()!=manifest['index_sha256'][name]:
            raise ValueError(f'{name} index checksum mismatch')
        if len(indices)!=manifest['counts'][name] or len(set(indices.tolist()))!=len(indices):
            raise ValueError(f'invalid {name} indices')
        if np.any(indices<0) or np.any(indices>=len(records)): raise ValueError('out-of-range index')
    if np.intersect1d(train,val).size: raise ValueError('train/validation overlap')
    users=sorted({int(r['userId']) for r in records}); items=sorted({int(r['movieId']) for r in records})
    um={u:i for i,u in enumerate(users)}; im={v:i for i,v in enumerate(items)}
    rows=np.fromiter((um[int(records[int(i)]['userId'])] for i in val),dtype=np.int64,count=len(val))
    cols=np.fromiter((im[int(records[int(i)]['movieId'])] for i in val),dtype=np.int64,count=len(val))
    y=np.fromiter((float(records[int(i)]['rating']) for i in val),dtype=float,count=len(val))
    train_cols=np.fromiter((im[int(records[int(i)]['movieId'])] for i in train),dtype=np.int64,count=len(train))
    warm=np.isin(cols,np.unique(train_cols))
    return rows,cols,y,warm,(len(users),len(items))


def score_validation(pred, truth, warm):
    pred=np.asarray(pred,dtype=float); truth=np.asarray(truth,dtype=float); warm=np.asarray(warm,dtype=bool)
    if pred.ndim!=1 or pred.shape!=truth.shape or pred.shape!=warm.shape or len(pred)==0:
        raise ValueError('invalid validation arrays')
    if not np.all(np.isfinite(pred)) or not np.all(np.isfinite(truth)):
        raise ValueError('nonfinite prediction or label')
    def rmse(mask):
        return float(np.sqrt(np.mean((pred[mask]-truth[mask])**2))) if mask.any() else None
    return {'validation_n':len(pred),'validation_rmse':rmse(np.ones(len(pred),dtype=bool)),
            'warm_item_n':int(warm.sum()),'warm_item_rmse':rmse(warm),
            'cold_item_n':int((~warm).sum()),'cold_item_rmse':rmse(~warm),
            'validation_clipped_rmse':float(np.sqrt(np.mean((np.clip(pred,0.5,5.0)-truth)**2)))}


def select_by_family(records, plan):
    """Only successful candidates eligible; stable listed-order tie break."""
    result={}
    for family, configs in plan['candidates'].items():
        eligible=[]
        for i,params in enumerate(configs):
            matching=[r for r in records if r.get('family')==family and r.get('params')==params]
            if len(matching)!=1: raise ValueError(f'missing or duplicate candidate: {family} {params}')
            r=matching[0]
            if r.get('status')=='ok':
                val=r.get('validation_rmse')
                if not isinstance(val,(int,float)) or not np.isfinite(val):
                    raise ValueError('invalid validation score')
                eligible.append((float(val),i,r))
        result[family]=min(eligible,key=lambda t:(t[0],t[1]))[2] if eligible else None
    return result
