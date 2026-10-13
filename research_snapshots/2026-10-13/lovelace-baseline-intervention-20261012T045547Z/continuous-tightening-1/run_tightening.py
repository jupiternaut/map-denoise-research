"""Public-only boundary refinement, HULL output, then non-decision probes."""
import time
from io_contract import *
import tightening_solver as solver
def selections(tasks,sets):
    lookup={(s['observation_id'],s['arm']):s for s in sets}
    return [{'task_id':t['task_id'],'observation_id':t['observation_id'],'arm':arm,
             **solver.select_depth(lookup[t['observation_id'],arm]['outer_intervals'],t['incumbent'],'HULL')}
            for t in tasks for arm in solver.MODES]
def main():
    target(); validate_old()
    frozen=load(OUT/'SOURCE_LOCK.json')['files']; locked=load(OUT/'INPUT_PRELOCK.json')['files']
    require(sources()==frozen and inputs()==locked,'prelocked bytes')
    budget=load(OUT/'BUDGET.json'); main_end=budget['main_deadline_epoch']
    observations,anchors,oldarms,baseline=public_inputs()
    sets=[]
    for i,photo in enumerate(observations):
        oid=photo['observation_id']; records=[]
        for arm in solver.MODES:
            solver.clear_geometry_cache()
            branches=[solver.solve_branch(photo,oldarms[oid]['box']['joint'],anchors[oid],oldbranch,arm,
                         deadline=time.monotonic()+max(0,min(6,main_end-time.time())))
                      for oldbranch in baseline[oid]['branches']]
            union=solver.merge_intervals(pair for b in branches for pair in b['outer_intervals'])
            records.append({'observation_id':oid,'arm':arm,'branches':branches,
                'outer_intervals':[[solver.encode(lo),solver.encode(hi)] for lo,hi in union],
                'hull':None if not union else [solver.encode(union[0][0]),solver.encode(union[-1][1])]})
        sets.extend(records); save(OUT/('checkpoint/'+oid+'.json'),records)
        status={'status':'RUNNING','observations_complete':i+1,'sets':len(sets),'seconds':time.time()-budget['start_epoch']}
        save(OUT/'RUN_STATUS.json',status,replace=True); print(json.dumps(status),flush=True)
    require(len(sets)==72,'all sets required'); save(OUT/'sets.json',sets)
    require(load(OUT/'sets.json')==sets,'set readback')
    # FIRST task parse, after every set is saved.
    tasks=load(OLD/'tasks.json')
    require(len(tasks)==324 and len({t['task_id'] for t in tasks})==324,'324 unique tasks')
    require(all(set(t)=={'task_id','base_task_id','observation_id','incumbent'} for t in tasks),'task schema')
    predictions=selections(tasks,sets); require(len(predictions)==648,'prediction matrix')
    save(OUT/'predictions.json',predictions)
    require(load(OUT/'predictions.json')==predictions,'prediction readback')
    save(OUT/'PREDICTION_RECEIPT.json',{'sets_sha256':sha(OUT/'sets.json'),'predictions_sha256':sha(OUT/'predictions.json'),
         'persisted_epoch':time.time(),'private_parsed':False,'probes_started':False})
    probes=[]; photos={p['observation_id']:p for p in observations}
    for record in (r for r in sets if r['arm']=='camera_tiles'):
        oid=record['observation_id']
        for pos in probe_positions(record,solver.exact):
            solver.anchor.clear_geometry_cache()
            result=solver.anchor.complete_branch(photos[oid],oldarms[oid]['box']['joint'],anchors[oid],'both',
                     solver.exact(pos['depth']),pos['branch'],deadline=time.monotonic()+max(0,min(.5,main_end-time.time())))
            probes.append({'observation_id':oid,**pos,'result':result})
    save(OUT/'probes.json',probes)
    require(sources()==frozen and inputs()==locked,'postmethod bytes')
    save(OUT/'RUN_STATUS.json',{'status':'COMPLETE','sets':72,'predictions':648,'probes':len(probes),
         'seconds':time.time()-budget['start_epoch'],'main_cutoff_reached':time.time()>=main_end},replace=True)
if __name__=='__main__':main()
