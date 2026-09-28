"""Small observation-to-decision core. Labels and evaluator are not imported."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

PARAMS = dict(loss='squared_error', learning_rate=.08, max_iter=80, max_leaf_nodes=7,
              max_depth=3, min_samples_leaf=80, l2_regularization=1.,
              random_state=20260926, early_stopping=False)
METHODS = ('independent_absolute','joint_common','joint_support','normalized_max','support_margin')
SETTINGS = ('balanced','native_priority','natural')

def contextual_features(features, geometry):
    """Return [N,2,2D+4]; exchanging proposals exchanges feature rows."""
    displacement = geometry[:,1:] - geometry[:,:1]
    lengths = np.linalg.norm(displacement,axis=2)
    separation = np.linalg.norm(geometry[:,1]-geometry[:,2],axis=1)
    product = lengths[:,0]*lengths[:,1]
    cosine = np.divide(np.sum(displacement[:,0]*displacement[:,1],axis=1), product,
                       out=np.zeros(len(features)), where=product>0)
    return np.stack([np.column_stack([features[:,c],features[:,1-c],lengths[:,c],
                      lengths[:,1-c],separation,cosine]) for c in range(2)],axis=1).astype(np.float32)

def case_weights(case_ids, e0):
    """Training-only weights for case-relative loss; never inference features."""
    result = np.empty(len(case_ids),float)
    for cid in np.unique(case_ids):
        idx = case_ids==cid
        result[idx] = 1/(idx.sum()*max(float(e0[idx].mean()),1e-6))
    return result/result.mean()

def fit_independent(features, gains, weights):
    return [HistGradientBoostingRegressor(**PARAMS).fit(features[:,c],gains[:,c],
                                                     sample_weight=weights) for c in range(2)]

def fit_shared(context, gains, weights):
    return HistGradientBoostingRegressor(**PARAMS).fit(
        context.reshape(-1,context.shape[-1]),gains.reshape(-1),sample_weight=np.repeat(weights,2))

def predict_shared(model, context):
    return model.predict(context.reshape(-1,context.shape[-1])).reshape(-1,2)

def canonical_scores(scores, geometry):
    """Identical physical choices cannot acquire two competing artificial utilities."""
    scores = np.asarray(scores,dtype=float).copy()
    assert scores.shape==(len(geometry),2)
    assert not np.isnan(scores).any() and not np.isposinf(scores).any()
    valid = np.linalg.norm(geometry[:,1:]-geometry[:,:1],axis=2)>1e-7
    same = np.linalg.norm(geometry[:,1]-geometry[:,2],axis=1)<=1e-7
    finite = np.isfinite(scores)
    mean = np.divide(np.where(finite,scores,0.).sum(axis=1),finite.sum(axis=1),
                     out=np.full(len(scores),-np.inf),where=finite.any(axis=1))
    scores[same] = mean[same,None]
    scores[~valid] = -np.inf
    return scores

def routes(scores, geometry, threshold=0., action='threshold'):
    """0=KEEP,1=A,2=B. Outputs—not index labels—are swap-equivariant."""
    if action=='keep':return np.zeros(len(geometry),np.uint8)
    values = canonical_scores(scores,geometry)
    a,b = values[:,0],values[:,1]
    choose_b = b>a
    ties = (a==b) & np.isfinite(a)
    ma = np.sum((geometry[:,1]-geometry[:,0])**2,axis=1)
    mb = np.sum((geometry[:,2]-geometry[:,0])**2,axis=1)
    choose_b |= ties & (mb<ma)
    remaining = ties & (mb==ma)
    for j in range(3):
        choose_b |= remaining & (geometry[:,2,j]<geometry[:,1,j])
        remaining &= geometry[:,2,j]==geometry[:,1,j]
    best = np.maximum(a,b)
    return np.where(np.isfinite(best)&(best>threshold),np.where(choose_b,2,1),0).astype(np.uint8)

def materialize(geometry, route):
    assert route.dtype==np.uint8 and route.shape==(len(geometry),)
    assert np.all(route<=2)
    return geometry[np.arange(len(geometry)),route]

def support_scores(features128):
    paired=features128[:,:,96:]
    return np.where(paired[:,:,29]>=.5,paired[:,:,24],-np.inf)
