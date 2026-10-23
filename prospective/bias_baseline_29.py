"""Train-only alternating ridge bias model, with explicit unseen-ID behavior."""
import numpy as np

def fit_bias(rows, cols, ratings, shape, *, regularization, max_iter=100, tol=1e-9):
    rows=np.asarray(rows); cols=np.asarray(cols); y=np.asarray(ratings,dtype=float)
    if len(shape)!=2 or any(int(x)!=x or x<1 for x in shape): raise ValueError('invalid shape')
    if rows.ndim!=1 or cols.ndim!=1 or y.ndim!=1 or not (len(rows)==len(cols)==len(y)) or not len(y): raise ValueError('invalid observations')
    if not (np.issubdtype(rows.dtype,np.integer) and np.issubdtype(cols.dtype,np.integer)): raise ValueError('indices must be integers')
    if np.any(rows<0) or np.any(rows>=shape[0]) or np.any(cols<0) or np.any(cols>=shape[1]) or not np.all(np.isfinite(y)): raise ValueError('invalid observations')
    if not np.isfinite(regularization) or regularization<=0 or max_iter<1 or not np.isfinite(tol) or tol<0: raise ValueError('invalid parameters')
    mu=float(np.mean(y)); u=np.zeros(shape[0]); v=np.zeros(shape[1]); history=[]
    nu=np.bincount(rows,minlength=shape[0]); nv=np.bincount(cols,minlength=shape[1])
    for iteration in range(max_iter):
        prevu=u.copy(); prevv=v.copy()
        u=np.bincount(rows,weights=y-mu-v[cols],minlength=shape[0])/(nu+regularization)
        v=np.bincount(cols,weights=y-mu-u[rows],minlength=shape[1])/(nv+regularization)
        residual=mu+u[rows]+v[cols]-y
        objective=float(np.dot(residual,residual)+regularization*(u@u+v@v))
        history.append(objective)
        delta=max(np.max(np.abs(u-prevu)),np.max(np.abs(v-prevv)))
        if delta<=tol: break
    return {'mean':mu,'user_bias':u,'item_bias':v,'iterations':len(history),'stop_reason':'absolute bias change' if delta<=tol else 'max_iter','objective_history':history}

def predict_bias(model, rows, cols):
    rows=np.asarray(rows);cols=np.asarray(cols)
    if rows.shape!=cols.shape or not np.issubdtype(rows.dtype,np.integer) or not np.issubdtype(cols.dtype,np.integer): raise ValueError('invalid indices')
    if np.any(rows<0) or np.any(cols<0) or np.any(rows>=len(model['user_bias'])) or np.any(cols>=len(model['item_bias'])): raise ValueError('out of bounds')
    return model['mean']+model['user_bias'][rows]+model['item_bias'][cols]
