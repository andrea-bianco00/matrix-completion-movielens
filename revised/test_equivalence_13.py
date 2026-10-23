"""Independent dense-vs-matrix-free SVP regression and algebra checks."""
import unittest
import numpy as np
from matrix_completion import svp
from sparse_svp import sparse_svp

class TestEquivalence13(unittest.TestCase):
    def test_multiple_seeds_ranks_patterns_and_steps(self):
        for seed in (4, 17, 81):
            rng = np.random.default_rng(seed)
            shape = (17, 13)
            y = rng.normal(size=shape)
            for density in (.25, .7, 1.):
                rows, cols = np.where(rng.random(shape) < density)
                for rank in (1, 3):
                    for step in (.4, 1.):
                        with self.subTest(seed=seed,density=density,rank=rank,step=step):
                            dense = svp(y,(rows,cols),rank=rank,step=step,max_iter=9,tol=0)
                            sparse = sparse_svp(shape,(rows,cols),y[rows,cols],rank=rank,step=step,max_iter=9,tol=0)
                            np.testing.assert_allclose(sparse.to_dense(),dense.matrix,atol=3e-7,rtol=3e-7)
                            np.testing.assert_allclose(sparse.train_rmse,dense.train_errors[:len(sparse.train_rmse)],atol=1e-8,rtol=1e-8)
                            self.assertLessEqual(sparse.iterations, 9)
    def test_low_rank_change_identity(self):
        rng=np.random.default_rng(923)
        for m,n,r in ((9,7,1),(13,8,3),(20,15,5)):
            a=rng.normal(size=(m,r)); b=rng.normal(size=(r,n))
            c=rng.normal(size=(m,r)); d=rng.normal(size=(r,n))
            ua,sa,vta=np.linalg.svd(a@b,full_matrices=False)
            uc,sc,vtc=np.linalg.svd(c@d,full_matrices=False)
            ua,sa,vta=ua[:,:r],sa[:r],vta[:r]
            uc,sc,vtc=uc[:,:r],sc[:r],vtc[:r]
            cross=np.sum((ua.T@uc)*(sa[:,None]*sc[None,:])*(vta@vtc.T))
            squared=max(0.,sum(sa**2)+sum(sc**2)-2*cross)
            actual=np.linalg.norm((ua*sa)@vta-(uc*sc)@vtc)**2
            np.testing.assert_allclose(squared,actual,atol=1e-10,rtol=1e-10)
    def test_prediction_on_holdout(self):
        rng=np.random.default_rng(18)
        y=rng.normal(size=(12,9)); rows,cols=np.where(rng.random(y.shape)<.5)
        result=sparse_svp(y.shape,(rows,cols),y[rows,cols],rank=2,max_iter=7)
        hr=np.array([0,1,3,6,10]); hc=np.array([2,4,5,8,1])
        np.testing.assert_allclose(result.predict(hr,hc),result.to_dense()[hr,hc],atol=1e-10)

if __name__=='__main__': unittest.main()
