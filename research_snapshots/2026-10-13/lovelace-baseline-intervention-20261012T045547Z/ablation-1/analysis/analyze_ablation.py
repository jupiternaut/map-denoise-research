"""Post-run exact contrasts only; never used to select masks or actions."""
from fractions import Fraction as Q
import csv
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'outputs/ablation-v1'
FULL=ROOT.parent/'recovery-1/outputs/baseline-v1'


def load(path):
    return json.loads(path.read_text())


def q(value):
    return Q(value['numerator'],value['denominator']) if isinstance(value,dict) else Q(value)


def select(rows,unit,method,condition,group,value=''):
    chosen=[r for r in rows if r['unit_id']==unit and r['method']==method and r['condition']==condition]
    if group=='all_shifted':
        return [r for r in chosen if q(r['initial_ae_mm'])!=0]
    if group=='correct':
        return [r for r in chosen if q(r['initial_ae_mm'])==0]
    if group=='negative_shifted':
        return [r for r in chosen if q(r['initial_ae_mm'])!=0 and r['case']=='equal_full_negative']
    return [r for r in chosen if q(r['initial_ae_mm'])!=0 and r['case']!='equal_full_negative' and
        (group=='informative_shifted' or group=='depth_informative' and r['depth_mm']==value or
         group=='case_informative' and r['case']==value)]


def score(rows):
    if not rows:
        raise ValueError('empty analysis group')
    mse=sum((q(r['se_mm2']) for r in rows),Q(0))/len(rows)
    return {'n':len(rows),'mse_mm2':str(mse),'mse_float':float(mse),
        'mae_mm':str(sum((q(r['ae_mm']) for r in rows),Q(0))/len(rows)),
        'improved':sum(q(r['actual_gain_mm2'])>0 for r in rows),
        'same':sum(q(r['actual_gain_mm2'])==0 for r in rows),
        'harmed':sum(q(r['actual_gain_mm2'])<0 for r in rows)}


def paired(left,right):
    lm={r['task_id']:r for r in left}
    rm={r['task_id']:r for r in right}
    if len(lm)!=len(left) or set(lm)!=set(rm):
        raise ValueError('unpaired analysis identities')
    differences=[q(lm[t]['se_mm2'])-q(rm[t]['se_mm2']) for t in lm]
    return {'n':len(lm),'better':sum(d>0 for d in differences),'same':sum(d==0 for d in differences),
        'worse':sum(d<0 for d in differences),
        'new_repairs':sum(q(lm[t]['actual_gain_mm2'])<=0<q(rm[t]['actual_gain_mm2']) for t in lm),
        'lost_repairs':sum(q(rm[t]['actual_gain_mm2'])<=0<q(lm[t]['actual_gain_mm2']) for t in lm),
        'mean_mse_gain_mm2':str(sum(differences,Q(0))/len(lm))}


def intervals(outer):
    return [tuple(map(q,value)) for value in outer['retained_intervals']]


def subset(inner,outer):
    return all(any(a<=left<=right<=b for a,b in outer) for left,right in inner)


def main():
    receipt=load(OUTPUT/'verification/VERIFY.json')
    if receipt['status']!='PASS' or receipt['output']!=str(OUTPUT):
        raise RuntimeError('analysis requires exact verified target')
    masked=load(OUTPUT/'results.json')
    reference=load(OUTPUT/'reference/results.json')
    replay=load(OUTPUT/'selectors/results.json')
    new=load(OUTPUT/'outers.json')
    full=load(FULL/'outers.json')
    depths=sorted({r['depth_mm'] for r in reference},key=Q)
    groups=[('all_shifted',''),('informative_shifted',''),('correct',''),('negative_shifted','')]+[('depth_informative',depth) for depth in depths]
    groups += [('case_informative',case) for case in sorted({r['case'] for r in reference if r['case']!='equal_full_negative'})]
    cells,contrasts,selector_pairs=[],[],[]
    for unit in ('baseline_70','baseline_80'):
        for group,value in groups:
            losses={}
            for method in ('calibrated','joint'):
                current=select(reference,unit,method,'full',group,value)
                losses[method,'full']=q(score(current)['mse_mm2'])
                cells.append({'unit_id':unit,'method':method,'condition':'full','group':group,'value':value,**score(current)})
                for condition in ('drop_F','matched_drop'):
                    rows=select(masked,unit,method,condition,group,value)
                    scored=score(rows)
                    losses[method,condition]=q(scored['mse_mm2'])
                    cells.append({'unit_id':unit,'method':method,'condition':condition,'group':group,'value':value,**scored})
                    contrasts.append({'kind':'deleted_vs_full','unit_id':unit,'method':method,'condition':condition,'group':group,'value':value,
                        'E_mm2':str(losses[method,condition]-losses[method,'full']),**paired(current,rows)})
                for selector in ('hybrid','midpoint','projection'):
                    selected=select(replay,unit,method,selector,group,value)
                    cells.append({'unit_id':unit,'method':method,'condition':'selector_'+selector,'group':group,'value':value,**score(selected)})
            contrasts.append({'kind':'bias_interaction','unit_id':unit,'group':group,'value':value,
                'full_cal_minus_joint_mm2':str(losses['calibrated','full']-losses['joint','full']),
                'drop_F_cal_minus_joint_mm2':str(losses['calibrated','drop_F']-losses['joint','drop_F']),
                'interaction_mm2':str((losses['calibrated','full']-losses['joint','full'])-(losses['calibrated','drop_F']-losses['joint','drop_F']))})
        for selector in ('hybrid','midpoint','projection'):
            for group,value in [('all','')]+groups:
                a=[r for r in replay if r['unit_id']==unit and r['condition']==selector and r['method']=='calibrated'] if group=='all' else select(replay,unit,'calibrated',selector,group,value)
                b=[r for r in replay if r['unit_id']==unit and r['condition']==selector and r['method']=='joint'] if group=='all' else select(replay,unit,'joint',selector,group,value)
                selector_pairs.append({'unit_id':unit,'selector':selector,'group':group,'value':value,**paired(a,b)})
    for method in ('calibrated','joint'):
        for group,value in groups:
            def find(unit,condition):
                return q(next(c['mse_mm2'] for c in cells if (c['unit_id'],c['method'],c['condition'],c['group'],c['value'])==(unit,method,condition,group,value)))
            e70=find('baseline_70','drop_F')-find('baseline_70','full')
            e80=find('baseline_80','drop_F')-find('baseline_80','full')
            contrasts.append({'kind':'baseline_fringe_interaction','method':method,'group':group,'value':value,
                'E70_mm2':str(e70),'E80_mm2':str(e80),'E80_minus_E70_mm2':str(e80-e70)})
    nesting=[]
    for oid,arms in new.items():
        for mask_name,methods in arms.items():
            for method,arm in methods.items():
                old=full[oid][method]['outer']
                nesting.append({'observation_id':oid,'mask_name':mask_name,'method':method,
                    'full_union_subset_deleted':subset(intervals(old),intervals(arm['outer'])),
                    'union_equal_full':intervals(old)==intervals(arm['outer']),
                    'hull_equal_full':old['hull']==arm['outer']['hull'],
                    'calibration_equal_full':arm['calibration']==full[oid][method]['calibration']})
    full_nesting=[]
    for oid,arms in full.items():
        hc,hj=arms['calibrated']['outer']['hull'],arms['joint']['outer']['hull']
        full_nesting.append({'observation_id':oid,'joint_union_subset_cal':subset(intervals(arms['joint']['outer']),intervals(arms['calibrated']['outer'])),
            'joint_hull_subset_cal':hj is None or hc is not None and q(hc[0])<=q(hj[0])<=q(hj[1])<=q(hc[1])})
    result={'verified_target':str(OUTPUT),'depths_mm':depths,'cells':cells,'contrasts':contrasts,'selector_cal_joint_pairs':selector_pairs,
        'masked_outer_nesting':nesting,'full_outer_nesting':full_nesting,
        'limitations':['Post-run mechanism contrasts; original masks and actions unchanged.',
            'Same paired synthetic images, not new independent scenes.', 'Matched deletion equalizes view/row/count, not information content.']}
    destination=ROOT/'analysis/CONTRASTS.json'
    with destination.open('x') as stream:
        json.dump(result,stream,indent=2)
        stream.write('\n')
    with (ROOT/'analysis/CELLS.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(cells[0]))
        writer.writeheader()
        writer.writerows(cells)
    print(json.dumps({'cells':len(cells),'contrasts':len(contrasts),'selector_pairs':len(selector_pairs),'nesting':len(nesting),'output':str(destination)}))


if __name__=='__main__':
    main()
