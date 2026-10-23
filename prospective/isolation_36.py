"""Create a verified, disjoint export. Never place held-out labels in the train directory."""
import csv, hashlib, io, json, zipfile
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
FIELDS=['userId','movieId','rating','timestamp']

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def export(source, destination):
    destination=Path(destination)
    if destination.exists() and any(destination.iterdir()): raise ValueError('destination must be empty')
    manifest=json.loads((HERE/'split_manifest_18.json').read_text())
    with zipfile.ZipFile(source) as z:
        names=[n for n in z.namelist() if n.endswith('/ratings.csv')]
        if len(names)!=1: raise ValueError('ambiguous source')
        raw=z.read(names[0])
    if hashlib.sha256(raw).hexdigest()!=manifest['source_sha256']: raise ValueError('source checksum mismatch')
    records=list(csv.DictReader(io.StringIO(raw.decode('utf-8'))))
    if len(records)!=manifest['source_rows']: raise ValueError('row count mismatch')
    if list(records[0])!=FIELDS: raise ValueError('unexpected fields')
    with np.load(HERE/'split_indices_18.npz') as data:
        parts={name:data[name].copy() for name in ('train','validation','test')}
    all_indices=[]
    for name, idx in parts.items():
        if len(idx)!=manifest['counts'][name]: raise ValueError('count mismatch')
        if hashlib.sha256(idx.astype('<i8').tobytes()).hexdigest()!=manifest['index_sha256'][name]: raise ValueError('index checksum mismatch')
        if np.any(idx<0) or np.any(idx>=len(records)): raise ValueError('invalid index')
        all_indices.extend(idx.tolist())
    if len(set(all_indices))!=len(records): raise ValueError('non-partitioned indices')
    users=sorted({int(r['userId']) for r in records})
    items=sorted({int(r['movieId']) for r in records})
    destination.mkdir(parents=True,exist_ok=True)
    (destination/'fit').mkdir(); (destination/'validation').mkdir(); (destination/'sealed_test').mkdir()
    catalog={'users':users,'items':items}
    (destination/'fit'/'catalog.json').write_text(json.dumps(catalog,separators=(',',':')))
    outputs={}
    for name, idx in parts.items():
        folder={'train':'fit','validation':'validation','test':'sealed_test'}[name]
        path=destination/folder/(name+'.csv')
        with path.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=FIELDS,lineterminator='\n'); writer.writeheader()
            for i in idx: writer.writerow(records[int(i)])
        outputs[name]={'count':len(idx),'sha256':sha(path)}
    metadata={'source_sha256':manifest['source_sha256'],'index_sha256':manifest['index_sha256'],
              'counts':manifest['counts'],'catalog_sha256':sha(destination/'fit'/'catalog.json'),
              'partitions':outputs,'isolation':'separate files; process filesystem access control required for strict enforcement'}
    (destination/'export_manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return metadata

def load_fit(fit_dir):
    """Only reads paths under fit_dir; does not take an original source archive."""
    fit_dir=Path(fit_dir)
    catalog=json.loads((fit_dir/'catalog.json').read_text())
    users=catalog['users']; items=catalog['items']
    if users!=sorted(set(users)) or items!=sorted(set(items)): raise ValueError('bad catalog')
    um={v:i for i,v in enumerate(users)}; im={v:i for i,v in enumerate(items)}
    with (fit_dir/'train.csv').open(newline='') as f: records=list(csv.DictReader(f))
    if len(records)!=70585: raise ValueError('unexpected training count')
    rows=np.array([um[int(r['userId'])] for r in records],dtype=np.int64)
    cols=np.array([im[int(r['movieId'])] for r in records],dtype=np.int64)
    y=np.array([float(r['rating']) for r in records],dtype=float)
    if not np.all(np.isfinite(y)): raise ValueError('nonfinite rating')
    return rows,cols,y,(len(users),len(items))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('destination');a=p.parse_args()
    print(json.dumps(export(a.source,a.destination),indent=2))
