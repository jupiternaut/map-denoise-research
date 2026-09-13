"""Conditional post-upstream cross-fitting of local candidate planes.

No ground truth is accepted. The upstream axis, bias, cells, original labels and
support were estimated on all observations. Thus this is NOT fully independent
cross-fitting of the whole reconstruction, nor a new independent observation.
"""
from pathlib import Path
import importlib.util
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('_v9_v7',ROOT/'exploration_v7/algorithm/reassociation.py')
V7=importlib.util.module_from_spec(spec);spec.loader.exec_module(V7)

def split_rows(scan_id,seed=913821):
    """Balanced random halves within each scan, independent of XYZ values."""
    scans=np.asarray(scan_id);fold=np.empty(len(scans),np.int64)
    rng=np.random.default_rng(seed)
    for sid in np.unique(scans):
        idx=np.flatnonzero(scans==sid);idx=idx[rng.permutation(len(idx))]
        fold[idx]=np.arange(len(idx))%2
    return fold

def candidate_search(state,original_artifacts,sigma_mm,budget=6,folds=2,split_seed=913821):
    """Return legal V8-style artifacts with fold-specific dictionaries.

For a matched one-fold control use folds=1. Training initialization and EM then
    use all active rows; full-rank two-fold initialization and EM use the opposite
    half. A deficient initialization explicitly falls back to a full-data seed.
Frozen original group membership and candidate access remain shared upstream.
Empty/rank-deficient training groups retain their previous legal seed; counted.
"""
    start=time.perf_counter();sigma=float(sigma_mm)
    if folds not in (1,2) or not isinstance(budget,int) or budget<0 or budget>100:
        raise ValueError('folds in {1,2}, integer budget in [0,100] required')
    if not np.isfinite(sigma) or sigma<=0:raise ValueError('positive sigma required')
    active=state['active'];original=state['order'][active];n=len(state['world'])
    x,y,w=state['design'][active],state['corrected'][active],state['weights'][active]
    seed_beta=original_artifacts['coefficients'];gcount=len(seed_beta)
    mask=original_artifacts['candidate_mask'][original]
    seed_groups=original_artifacts['group_ids'][original]
    held=split_rows(state['scan_id'][original],split_seed) if folds==2 else np.zeros(len(active),np.int64)
    beta_all=[];query_groups=np.full(n,-1,np.int64)
    candidate=np.zeros((n,gcount*folds),bool);resp=np.zeros((n,gcount*folds))
    fold_world=np.full(n,-1,np.int64);fold_world[original]=held
    train_masks=[];query_masks=[];diagnostics=[];counter=V7._counter()
    for fold in range(folds):
        query=held==fold;train=held!=fold if folds==2 else np.ones(len(active),bool)
        beta=seed_beta.copy();fallback=[];initial_ranks=[]
        for group in range(gcount):
            take=train&(seed_groups==group)
            if take.sum()>=3:
                value,record=V7._wls(x[take],y[take],w[take],counter)
                rank=record['rank']
            else:rank=0
            initial_ranks.append(rank)
            if rank==3:beta[group]=value
            else:fallback.append(group)
        trace=[]
        for step in range(budget):
            r,_,_=V7._e_step(x[train],y[train],beta,mask[train],sigma,counter)
            beta,fit=V7._m_step(x[train],y[train],w[train],r,mask[train],beta,counter)
            trace.append([record['rank'] for record in fit])
        r,_,_=V7._e_step(x[query],y[query],beta,mask[query],sigma,counter)
        assigned,_,_=V7._hard_assignment(r,mask[query],seed_groups[query])
        rows=original[query];sl=slice(fold*gcount,(fold+1)*gcount)
        query_groups[rows]=assigned+fold*gcount;candidate[rows,sl]=mask[query];resp[rows,sl]=r
        beta_all.extend(beta)
        tr=np.zeros(n,bool);tr[original[train]]=True;qr=np.zeros(n,bool);qr[rows]=True
        train_masks.append(tr);query_masks.append(qr)
        diagnostics.append(dict(fold=fold,train_count=int(train.sum()),query_count=int(query.sum()),
                train_query_overlap=int(np.sum(train&query)),initial_fit_rank=initial_ranks,
                initial_seed_fallback_groups=fallback,em_fit_ranks=trace))
    support=np.zeros(n,bool);support[state['order']]=state['support']
    arrays=dict(group_ids=query_groups,candidate_mask=candidate,responsibility=resp,
                coefficients=np.asarray(beta_all),support_mask=support,fold_id=fold_world,
                candidate_training_mask=np.asarray(train_masks),candidate_query_mask=np.asarray(query_masks))
    info=dict(folds=folds,budget=budget,split_seed=split_seed,seconds=time.perf_counter()-start,
       fold_diagnostics=diagnostics,training_work=counter,truth_fields_used=[],
       scope='conditional cross-fit only after full-data upstream; not independent full-pipeline fit',
       changed_from_v7='training-row original-group WLS initialization plus fold-specific EM and dictionary',
       dictionary_groups_per_fold=gcount,stored_parameter_count=3*gcount*folds,
       fallback_count=sum(len(d['initial_seed_fallback_groups']) for d in diagnostics))
    return arrays,info

def project_candidate(state,artifacts):
    """Project to chosen fitted candidate; no posterior averaging or hard refit."""
    start=time.perf_counter();active=state['active'];original=state['order'][active]
    group=artifacts['group_ids'][original]
    height=state['local'][:,2].copy()
    height[active]=np.einsum('ij,ij->i',state['design'][active],artifacts['coefficients'][group])
    take=np.flatnonzero(state['support']);rows=state['order'][take];out=state['world'].copy()
    out[rows]+=(height[take]-state['local'][take,2])[:,None]*state['normal']/1000.
    assert np.isfinite(out).all()
    np.testing.assert_array_equal(out[~artifacts['support_mask']],state['world'][~artifacts['support_mask']])
    return out,dict(mode='candidate_map',seconds=time.perf_counter()-start,truth_fields_used=[]),dict(
        group_ids=artifacts['group_ids'].copy(),support_mask=artifacts['support_mask'].copy(),
        output_candidate_coefficients=artifacts['coefficients'].copy())
