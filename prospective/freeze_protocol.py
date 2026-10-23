"""Freeze prospective benchmark splits; never read held-out ratings into model selection."""
import csv, hashlib, io, json, zipfile
from pathlib import Path
import numpy as np
BASE=Path(__file__).resolve().parent
SOURCE=Path('/mnt/data/Dimension Reduction and Matrix Completion.zip')
SEED=20261024
with zipfile.ZipFile(SOURCE) as z:
    names=[p for p in z.namelist() if p.endswith('/ratings.csv')]
    if len(names)!=1: raise ValueError('Expected exactly one ratings.csv')
    raw=z.read(names[0])
records=list(csv.DictReader(io.StringIO(raw.decode('utf-8'))))
assert len(records)==100836
pairs=[(int(x['userId']),int(x['movieId'])) for x in records]
assert len(set(pairs))==len(pairs)
assert all(np.isfinite(float(x['rating'])) for x in records)
# A different, explicitly prospective split; no test values consulted in this script.
perm=np.random.default_rng(SEED).permutation(len(records))
a=int(.7*len(records)); b=int(.85*len(records))
parts={'train':perm[:a], 'validation':perm[a:b], 'test':perm[b:]}
assert sum(map(len,parts.values()))==len(records)
assert np.array_equal(np.sort(np.concatenate(list(parts.values()))),np.arange(len(records)))
np.savez_compressed(BASE/'split_indices_18.npz', **parts)
# Hashes enable later checks that index arrays were not silently replaced.
manifest={
 'source_sha256':hashlib.sha256(raw).hexdigest(),
 'source_rows':len(records),
 'seed':SEED,
 'split_method':'numpy.random.default_rng(seed).permutation(n), contiguous 70/15/15 slices; floor boundaries',
 'counts':{k:len(v) for k,v in parts.items()},
 'index_sha256':{k:hashlib.sha256(v.astype('<i8').tobytes()).hexdigest() for k,v in parts.items()},
 'rating_scale':sorted(set(float(x['rating']) for x in records)),
 'data_policy':'Only training values for fitting; validation for tuning; test indices and ratings inaccessible to model-selection code.'
}
(BASE/'split_manifest_18.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({k:manifest[k] for k in ('source_sha256','seed','counts','index_sha256')},indent=2))
