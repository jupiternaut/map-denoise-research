"""Export one post-hoc two-depth explanatory certificate; not a repair selector."""
import json,sys
from fractions import Fraction as Q
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import verify_pose as v
OUT=ROOT/'outputs/pose-v1'
DEST=ROOT/'diagnostic-audit/PAIR_WITNESS.json'
def main():
    assert not DEST.exists()
    assert v.load(OUT/'STAGE_STATUS.json')['status']=='INCOMPLETE'
    assert v.load(ROOT/'diagnostic-audit/SAVED_RECORD_AUDIT.json')['status']=='SAVED_RECORD_NUMERICAL_PASS'
    rows=v.load(OUT/'results.json');records={r['task_id']:r for r in v.load(OUT/'diagnoses.json')}
    eligible=[r for r in rows if r['variation_index']==1 and r['truth_mm']=='9007/14' and r['case']!='equal_full_negative' and Q(r['initial_ae_mm'])==3 and r['status']=='FORWARD_WITNESS']
    row=sorted(eligible,key=lambda r:r['task_id'])[0];record=records[row['task_id']]
    b=next(c for c in record['branches'] if c['witness'])
    publics={p['observation_id']:p for p in v.load(OUT/'observations.json')}
    public=publics[row['observation_id']];arm=v.load(v.A5_OUTPUT/'outers.json')[row['observation_id']]['box']['joint']
    # Private export follows the already independently checked full saved-public gate.
    facts={f['observation_id']:f for f in v.load(v.DATA/'private/truth.json')}
    fact=facts[row['observation_id']]
    actual={'offsets':[v.encoded(v.exact(x)) for x in fact['actual_offsets']],
            'darkness':[v.encoded(1-v.exact(x)) for x in fact['view_colors']],
            't':[v.encoded(v.BETA-v.exact(x)) for x in fact['actual_view_bias']],
            'noise':[v.encoded(v.exact(x)) for x in fact['image_noise']]}
    za,zb=v.exact(fact['truth']),v.exact(record['incumbent'])
    v.verify_forward(public,arm,za,fact['rear_branch'],actual,'certificate world A')
    v.verify_forward(public,arm,zb,b['branch'],b['witness'],'certificate world B')
    pair={'status':'TWO_FORWARD_EXPLANATIONS_VERIFIED','formal_stage_status':'INCOMPLETE','diagnostic_only':True,
          'selection':'post-hoc first lexicographic primary positive with 3mm separation; never used by the method',
          'task_id':row['task_id'],'public':public,'calibration_arm':arm,
          'world_a':{'depth':v.encoded(za),'branch':fact['rear_branch'],'witness':actual},
          'world_b':{'depth':v.encoded(zb),'branch':b['branch'],'witness':b['witness']},
          'depth_separation_mm':str(abs(za-zb)),
          'conditional_minimax_absolute_error_mm':str(abs(za-zb)/2),
          'guarantee_scope':'frozen public images and calibration intervals, all original parameter bounds; not an independent real-scene performance theorem'}
    with DEST.open('x') as f:json.dump(pair,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'path':str(DEST),'task':row['task_id'],'separation_mm':str(abs(za-zb)),'verified_worlds':2,'formal_stage_status':'INCOMPLETE'}))
if __name__=='__main__':main()
