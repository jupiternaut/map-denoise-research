"""One public A11 batch. Incumbents are parsed only after all supports persist."""
import time
from a11_io_contract import *
import accumulation_solver as solver
def arm_order(i): return solver.ARMS if i%2==0 else tuple(reversed(solver.ARMS))
def selections(tasks,sets):
    lookup={(s['observation_id'],s['arm']):s for s in sets}
    return [{'task_id':t['task_id'],'observation_id':t['observation_id'],'arm':arm,
        **solver.select_depth(lookup[t['observation_id'],arm]['outer_intervals'],t['incumbent'],'HULL')}
        for t in tasks for arm in solver.ARMS]
def main():
    target();validate_old()
    frozen=load(OUT/'SOURCE_LOCK.json')['files'];locked=load(OUT/'INPUT_PRELOCK.json')['files']
    require(sources()==frozen and inputs()==locked,'initial source/input lock')
    budget=load(OUT/'BUDGET.json');end=budget['main_deadline_epoch']
    photos,anchors,oldarms,source=public_inputs();sets=[]
    for i,photo in enumerate(photos):
        oid=photo['observation_id'];records=[]
        for arm in arm_order(i):
            solver.clear_geometry_cache()
            branches=[solver.solve_branch(photo,oldarms[oid]['box']['joint'],anchors[oid],b,arm,
                deadline=time.monotonic()+max(0,min(6,end-time.time()))) for b in source[oid]['branches']]
            union=solver.merge_intervals(pair for b in branches for pair in b['outer_intervals'])
            records.append({'observation_id':oid,'arm':arm,'branches':branches,
                'outer_intervals':[[solver.encode(lo),solver.encode(hi)] for lo,hi in union],
                'hull':None if not union else [solver.encode(union[0][0]),solver.encode(union[-1][1])]})
        sets.extend(records);save(OUT/('checkpoint/'+oid+'.json'),records)
        status={'status':'RUNNING','observations_complete':i+1,'sets':len(sets),'seconds':time.time()-budget['start_epoch']}
        save(OUT/'RUN_STATUS.json',status,replace=True);print(json.dumps(status),flush=True)
    require(len(sets)==72,'complete support matrix');save(OUT/'sets.json',sets)
    require(load(OUT/'sets.json')==sets,'support readback')
    tasks=load(OLD9/'tasks.json')
    require(len(tasks)==324 and len({t['task_id'] for t in tasks})==324,'unique task matrix')
    require(all(set(t)=={'task_id','base_task_id','observation_id','incumbent'} for t in tasks),'task schema')
    predictions=selections(tasks,sets);require(len(predictions)==648,'complete action matrix')
    save(OUT/'predictions.json',predictions);require(load(OUT/'predictions.json')==predictions,'action readback')
    save(OUT/'PREDICTION_RECEIPT.json',{'sets_sha256':sha(OUT/'sets.json'),
        'predictions_sha256':sha(OUT/'predictions.json'),'persisted_epoch':time.time(),'private_parsed':False})
    require(sources()==frozen and inputs()==locked,'postmethod lock')
    save(OUT/'RUN_STATUS.json',{'status':'COMPLETE','sets':72,'predictions':648,
        'seconds':time.time()-budget['start_epoch'],'main_cutoff_reached':time.time()>=end},replace=True)
if __name__=='__main__':main()
