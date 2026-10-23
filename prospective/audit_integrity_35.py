"""Read-only validation-results audit. Never reads held-out test ratings."""
import hashlib, json, math
from pathlib import Path
HERE=Path(__file__).resolve().parent
SOURCE=Path('/mnt/data/Dimension Reduction and Matrix Completion.zip')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def verify():
    freeze=json.loads((HERE/'freeze_34.json').read_text()); plan=json.loads((HERE/'benchmark_plan_28.json').read_text()); data=json.loads((HERE/'validation_results_34.json').read_text())
    paths={'plan':HERE/'benchmark_plan_28.json','runner':HERE/'runner_33.py','validation':HERE/'validation_32.py','source':SOURCE}
    assert all(sha(path)==freeze['sha256'][name] for name,path in paths.items()),'frozen code or source changed'
    assert data['freeze_sha256']==sha(HERE/'freeze_34.json'),'freeze pointer mismatch'
    assert data['complete'] is True
    records=data['records']; expected=[(fam,p) for fam,ps in plan['candidates'].items() for p in ps]
    assert len(records)==len(expected)==11
    for fam,params in expected:
        matches=[r for r in records if r.get('family')==fam and r.get('params')==params]
        assert len(matches)==1,(fam,params)
        r=matches[0]
        assert r['status']=='ok' and r['evaluated_split']=='train_and_validation'
        assert r['training_n']==70585 and r['validation_n']==15125
        assert r['warm_item_n']+r['cold_item_n']==r['validation_n']
        assert r['shape']==[610,9724]
        assert all(math.isfinite(r[k]) and r[k]>=0 for k in ('training_rmse','validation_rmse','validation_clipped_rmse'))
        # Check weighted RMSE decomposition to detect inconsistent subgroup metrics.
        w=r['warm_item_n']; c=r['cold_item_n']; n=r['validation_n']
        if w and c:
            recomputed=math.sqrt((w*r['warm_item_rmse']**2+c*r['cold_item_rmse']**2)/n)
            assert math.isclose(recomputed,r['validation_rmse'],rel_tol=1e-11,abs_tol=1e-11)
    from validation_32 import select_by_family
    selected=select_by_family(records,plan)
    for family,r in selected.items():
        assert r==data['selected'][family],f'selection mismatch {family}'
    overall=min(selected.values(),key=lambda r:r['validation_rmse'])
    return {'status':'verified','n_candidates':len(records),'selected':{f:{'params':r['params'],'validation_rmse':r['validation_rmse']} for f,r in selected.items()},'overall_selected':{'family':overall['family'],'params':overall['params'],'validation_rmse':overall['validation_rmse']},'validation_sha256':sha(HERE/'validation_results_34.json'),'freeze_sha256':sha(HERE/'freeze_34.json'),'test_labels_read':False}
if __name__=='__main__':
    out=verify(); (HERE/'integrity_result_35.json').write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
