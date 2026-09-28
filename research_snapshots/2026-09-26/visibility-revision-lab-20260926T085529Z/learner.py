"""Fixed six-arm development search; observation-only inference and shared units."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from common import routes, case_weights, old_router

FAMILIES=('base','geometry','interaction')
CAPACITIES=('shallow','rich')
ARMS=tuple(f'{family}_{capacity}' for family in FAMILIES for capacity in CAPACITIES)
POLICIES=('balanced','native_priority','recovery','natural')

def features_for(base,extra,arm):
    from visibility_features import GEOMETRY_WIDTH, FEATURE_NAMES
    assert base.ndim==3 and base.shape[1:]==(2,96)
    assert extra.shape==(len(base),2,len(FEATURE_NAMES))
    family,capacity=arm.rsplit('_',1)
    assert family in FAMILIES and capacity in CAPACITIES
    width={'base':0,'geometry':GEOMETRY_WIDTH,'interaction':len(FEATURE_NAMES)}[family]
    x=np.concatenate((base,extra[:,:,:width]),axis=2)
    assert np.isfinite(x).all()
    return x

def parameters(arm):
    rich=arm.endswith('_rich')
    return dict(loss='squared_error',learning_rate=.08,max_iter=160 if rich else 80,
                max_leaf_nodes=31 if rich else 7,max_depth=6 if rich else 3,
                min_samples_leaf=40 if rich else 80,l2_regularization=1.,
                random_state=20260926,early_stopping=False)

def fit_pair(x,gains,weights,arm):
    return [HistGradientBoostingRegressor(**parameters(arm)).fit(x[:,i],gains[:,i],
                sample_weight=weights) for i in range(2)]

def predict_pair(models,x):
    return np.column_stack([models[i].predict(x[:,i]) for i in range(2)])

def loss_row(d,geometry,scores,threshold=0.,action='threshold'):
    choice=routes(scores,geometry,threshold,action)
    after=d['losses'][np.arange(len(choice)),choice]
    relative=[];native=[];baseline=[];injected=[];accepted=[]
    for cid in np.unique(d['case_id']):
        mask=(d['case_id']==cid)&d['calibration_support']
        before=float(d['losses'][mask,0].mean());mse=float(after[mask].mean())
        relative.append(mse/max(before,1e-6));accepted.append(float((choice[mask]!=0).mean()))
        if d['condition'][mask][0]=='native':native.append(mse);baseline.append(before)
        else:injected.append(relative[-1])
    return dict(threshold=float(threshold),action=action,
        objective_relative_MSE=float(np.mean(relative)),native_MSE=float(np.mean(native)),
        native_identity_MSE=float(np.mean(baseline)),injected_relative_MSE=float(np.mean(injected)),
        accepted_fraction=float(np.mean(accepted)))

def calibration_key(row,policy):
    objective='injected_relative_MSE' if policy=='recovery' else 'objective_relative_MSE'
    return (row[objective],row['accepted_fraction'],-row['threshold'])

def calibrate(d,geometry,scores):
    best=old_router.canonical_scores(scores,geometry).max(axis=1)
    values=best[d['calibration_support']&np.isfinite(best)]
    thresholds=sorted(set([0.]+list(np.quantile(values,np.linspace(0,1,100))))) if len(values) else [0.]
    table=[loss_row(d,geometry,scores,t) for t in thresholds]
    keep=loss_row(d,geometry,scores,0.,'keep');table.append(keep)
    valid=[r for r in table if r['native_MSE']<=r['native_identity_MSE']+1e-12
           and r['injected_relative_MSE']<=.95+1e-12]
    result={p:dict(min(table,key=lambda r:calibration_key(r,p)),feasible=True)
            for p in ('balanced','recovery')}
    result['native_priority']=dict(min(valid,key=lambda r:calibration_key(r,'native_priority')),
        feasible=True) if valid else dict(keep,feasible=False)
    result['natural']=dict(loss_row(d,geometry,scores,0.),feasible=True)
    return result,table

def choose_winners(thresholds):
    result={}
    for policy in POLICIES:
        eligible=[arm for arm in ARMS if thresholds[arm][policy]['feasible']]
        if not eligible:eligible=list(ARMS)
        arm=min(eligible,key=lambda a:(*calibration_key(thresholds[a][policy],policy),ARMS.index(a)))
        result[policy]=dict(arm=arm,route_arm=arm+'__'+policy,**thresholds[arm][policy])
    return result
