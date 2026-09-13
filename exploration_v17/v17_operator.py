"""Adaptive gate complexity for parallel-surface mixture regression.

Public estimation receives measured X, h, sigma only. No evaluator imports.
"""
from pathlib import Path
import sys,time
import numpy as np
from scipy.special import log_expit,logsumexp,expit
from scipy.optimize import minimize

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'exploration_v16'),str(ROOT/'exploration_v15'),str(ROOT/'exploration_v11')]
from models import fit as old_fit
from spatial_filter import solve as legacy_solve,geometry

FAMILIES=('S','C','L','R')
PARAMETERS=dict(S=3,C=5,L=7,R=16)
METHODS=('old_constant64','old_spatial36','constant_only','linear_only','rbf_only','adaptive_bic','adaptive_cv')
RIDGE=.1


def feature_matrix(u,family,center,scale):
    z=(u-center)/scale
    if family=='C':return np.ones((len(u),1))
    linear=np.c_[np.ones(len(u)),z]
    if family=='L':return linear
    if family!='R':raise ValueError(family)
    anchors=np.array([(i,j) for i in (-.75,0,.75) for j in (-.75,0,.75)])
    rbf=np.exp(-np.sum((z[:,None,:]-anchors[None,:,:])**2,axis=2)/(2*.6**2))
    return np.c_[linear,rbf]


def objective(t,x,y,f,ridge=RIDGE):
    pred=t[:2][None,:]+(x@t[2:4])[:,None]
    residual=pred-y[:,None]
    logits=f@t[4:]
    lp=-.5*residual**2+np.c_[log_expit(-logits),log_expit(logits)]
    norm=logsumexp(lp,axis=1)
    r=np.exp(lp-norm[:,None])
    gr=r*residual
    penalty=np.zeros(f.shape[1]);penalty[1:]=ridge
    grad=np.r_[gr.sum(0),x.T@gr.sum(1),f.T@(expit(logits)-r[:,1])+penalty*t[4:]]
    return float(-norm.sum()+.5*np.dot(penalty,t[4:]**2)),grad


def gate_start(f,target):
    p=np.clip(np.mean(target),1e-8,1-1e-8)
    beta=np.zeros(f.shape[1]);beta[0]=np.log(p/(1-p))
    penalty=np.zeros(f.shape[1]);penalty[1:]=RIDGE
    for _ in range(8):
        q=expit(f@beta);w=np.maximum(q*(1-q),1e-6)
        h=f.T@(f*w[:,None])+np.diag(penalty+1e-9)
        step=np.linalg.solve(h,f.T@(q-target)+penalty*beta)
        step*=min(1.,3./max(np.linalg.norm(step),1e-12))
        beta-=step;beta[0]=np.clip(beta[0],-30.,30.)
    return beta


def training_seeds(x,y,sigma,xc,xs,ym):
    xx=(x[:,1:]-xc)/xs;yy=(y-ym)/sigma;design=np.c_[np.ones(len(y)),xx]
    old=legacy_solve(x,y,sigma,spatial=False,force_k=2,iterations=324)
    first=np.r_[(old['means']+old['slope']@xc-ym)/sigma,old['slope']*xs/sigma]
    pred=first[:2][None,:]+(xx@first[2:])[:,None]
    logits=old['gate'][0]
    lp=-.5*(yy[:,None]-pred)**2+np.array([log_expit(-logits),log_expit(logits)])
    prob=np.exp(lp-logsumexp(lp,axis=1)[:,None])[:,1]
    seeds=[dict(geometry=first,target=prob,constant_gate=float(logits))]
    one=np.linalg.lstsq(design,yy,rcond=1e-12)[0];residual=yy-design@one
    rng=np.random.default_rng(9150001)
    for j in range(63):
        if j<9:
            p=expit((residual-np.quantile(residual,(j+1)/10))*(1+j%3))
        else:
            feature=np.c_[residual,xx]@rng.normal(size=3)
            feature=(feature-np.quantile(feature,rng.uniform(.1,.9)))/max(feature.std(),1e-6)
            p=expit(feature*rng.choice([1.,3.,8.]))
        mu,a=geometry(design,yy,np.c_[1-p,p],None)
        mean=np.clip(p.mean(),1e-8,1-1e-8)
        seeds.append(dict(geometry=np.r_[mu,a],target=p,constant_gate=float(np.log(mean/(1-mean)))))
    return seeds


def evaluate(m,x,y,sigma):
    if m['k']==1:
        pred=x@m['plane']
        return dict(prediction=pred,groups=np.zeros(len(x),int),posterior=np.ones((len(x),1)),
                    log_density=-.5*((y-pred)/sigma)**2)
    f=feature_matrix(x[:,1:],m['family'],m['feature_center'],m['feature_scale'])
    logits=f@m['gate'];pred=m['means'][None,:]+(x[:,1:]@m['slope'])[:,None]
    lp=-.5*((y[:,None]-pred)/sigma)**2+np.c_[log_expit(-logits),log_expit(logits)]
    norm=logsumexp(lp,axis=1);posterior=np.exp(lp-norm[:,None]);g=np.argmax(lp,axis=1)
    return dict(prediction=pred[np.arange(len(x)),g],groups=g,posterior=posterior,log_density=norm)


def solve_family(x,y,sigma,family,seeds,xc,xs,ym,center,scale,warm=()):
    xx=(x[:,1:]-xc)/xs;yy=(y-ym)/sigma;f=feature_matrix(x[:,1:],family,center,scale)
    selected=seeds if family=='C' else seeds[:12]
    starts=[np.r_[s['geometry'],np.array([s['constant_gate']]) if family=='C' else gate_start(f,s['target'])] for s in selected]
    for m in warm:
        geom=np.r_[(m['means']+m['slope']@xc-ym)/sigma,m['slope']*xs/sigma]
        beta=np.zeros(f.shape[1]);beta[:len(m['gate'])]=m['gate']
        starts.append(np.r_[geom,beta])
    best=None;best_value=np.inf;records=[];start_min=np.inf
    for initial in starts:
        value=objective(initial,xx,yy,f)[0];start_min=min(start_min,value)
        if value<best_value:best=initial.copy();best_value=value
        opt=minimize(objective,initial,args=(xx,yy,f),jac=True,method='L-BFGS-B',
                     bounds=[(None,None)]*4+[(-30.,30.)]+[(None,None)]*(f.shape[1]-1),
                     options=dict(maxiter=324,ftol=1e-12,gtol=1e-8,maxls=40))
        value,grad=objective(opt.x,xx,yy,f)
        records.append(dict(initial=initial.copy(),final=opt.x.copy(),success=bool(opt.success),iterations=int(opt.nit),evaluations=int(opt.nfev),
                            objective=float(value),gradient_inf=float(np.max(abs(grad))),message=str(opt.message)))
        if np.isfinite(value) and value<best_value:best=opt.x.copy();best_value=value
    slope=best[2:4]*sigma/xs;means=best[:2]*sigma+ym-slope@xc;gate=best[4:].copy()
    order=np.argsort(means,kind='stable')
    if order[0]==1:means=means[::-1].copy();gate=-gate
    m=dict(k=2,family=family,means=means,slope=slope,gate=gate,feature_center=center,feature_scale=scale,
           starts=len(starts),solver_diagnostics=records,minimum_start_objective=start_min,final_objective=float(best_value),
           parameter_count=PARAMETERS[family])
    ev=evaluate(m,x,y,sigma);m.update(ev)
    m['nll']=float(-ev['log_density'].sum());m['ridge_penalty']=float(RIDGE*np.sum(gate[1:]**2))
    m['bic']=float(2*m['nll']+m['ridge_penalty']+m['parameter_count']*np.log(len(y)))
    return m


def fit_pool(x,y,sigma):
    x,y=np.asarray(x,float),np.asarray(y,float)
    if x.shape!=(len(y),3) or len(y)<12 or not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isfinite(sigma) or sigma<=0:
        raise ValueError('finite N x 3 affine design, height, positive sigma; N>=12')
    np.testing.assert_array_equal(x[:,0],1.)
    xc=x[:,1:].mean(0);xs=np.maximum(x[:,1:].std(0),1e-8);ym=float(y.mean())
    center=np.median(x[:,1:],axis=0);scale=np.maximum(np.ptp(x[:,1:],axis=0)/2,1e-6)
    plane=np.linalg.lstsq(x,y,rcond=1e-12)[0]
    s=dict(k=1,family='S',plane=plane,means=plane[:1],slope=plane[1:],gate=np.zeros(1),
           feature_center=center,feature_scale=scale,parameter_count=3,starts=1,solver_diagnostics=[])
    s.update(evaluate(s,x,y,sigma));s.update(nll=float(-s['log_density'].sum()),ridge_penalty=0.)
    s['bic']=float(2*s['nll']+3*np.log(len(y)))
    seeds=training_seeds(x,y,sigma,xc,xs,ym)
    pool={'S':s}
    for family,warm in [('C',()),('L',('C',)),('R',('C','L'))]:
        pool[family]=solve_family(x,y,sigma,family,seeds,xc,xs,ym,center,scale,tuple(pool[w] for w in warm))
    return pool


def folds_from_x(x,fold_count=3):
    order=np.lexsort((x[:,2],x[:,1]));rng=np.random.default_rng(9170001)
    fold=np.empty(len(x),int);fold[order[rng.permutation(len(x))]]=np.arange(len(x))%fold_count
    return fold


def cross_validation(x,y,sigma):
    folds=folds_from_x(x);scores={f:0. for f in FAMILIES};details=[]
    per_point={f:np.empty(len(y)) for f in FAMILIES}
    for fold in range(3):
        train=folds!=fold;valid=~train
        pool=fit_pool(x[train],y[train],sigma)
        record=dict(fold=fold,train_indices=np.flatnonzero(train),validation_indices=np.flatnonzero(valid),families={})
        for family,m in pool.items():
            losses=-evaluate(m,x[valid],y[valid],sigma)['log_density']
            per_point[family][valid]=losses;scores[family]+=float(losses.sum())
            record['families'][family]=dict(validation_nll=float(losses.sum()),train_nll=m['nll'],bic=m['bic'],
                means=m['means'],slope=m['slope'],gate=m['gate'],feature_center=m['feature_center'],feature_scale=m['feature_scale'],
                plane=m.get('plane'),k=m['k'],family=family,starts=m['starts'],
                solver_diagnostics=m['solver_diagnostics'],
                nonconverged_starts=sum(not d['success'] for d in m['solver_diagnostics']))
        details.append(record)
    return dict(scores=scores,per_point=per_point,folds=folds,details=details,
                rule='min total held-out Gaussian-mixture NLL, constants common to families omitted')


def choose(scores,families=FAMILIES):
    return min(families,key=lambda f:(round(scores[f],10),PARAMETERS[f],FAMILIES.index(f)))


def construct(x,y,sigma,with_cv=True):
    started=time.perf_counter();pool=fit_pool(x,y,sigma)
    cv=cross_validation(x,y,sigma) if with_cv else None
    scores={f:m['bic'] for f,m in pool.items()}
    choices=dict(constant_only=choose(scores,('S','C')),linear_only=choose(scores,('S','L')),
                 rbf_only=choose(scores,('S','R')),adaptive_bic=choose(scores))
    if cv is not None:choices['adaptive_cv']=choose(cv['scores'])
    outputs={method:dict(pool[family],selected_family=family,selection_method=method,truth_fields_used=[]) for method,family in choices.items()}
    # Frozen old baselines are independently run; no oracle slope supplied.
    outputs['old_constant64']=old_fit(x,y,sigma,'constant_free')
    outputs['old_spatial36']=old_fit(x,y,sigma,'spatial_free')
    return outputs,dict(pool=pool,cv=cv,choices=choices,seconds=time.perf_counter()-started,
                        input_fields=['measured_design','measured_height','supplied_sigma'],truth_fields_used=[])
