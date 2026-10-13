"""Read-only A4 result analysis after independent verification; exact paired accounts."""
from collections import Counter,defaultdict
from fractions import Fraction as Q
import argparse
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
OUTPUT=ROOT/'outputs/appearance-v1'
FULL=ROOT.parent/'recovery-1/outputs/baseline-v1'
METHODS=('calibrated','joint')


def load(path):
    return json.loads(Path(path).read_text())


def exact(value):
    if isinstance(value,dict):
        if set(value)!={'numerator','denominator'} or type(value['numerator'])is not int or type(value['denominator'])is not int or value['denominator']<=0:
            raise ValueError('invalid exact encoding')
        return Q(value['numerator'],value['denominator'])
    if isinstance(value,(bool,float)):
        raise ValueError('inexact metric')
    return Q(value)


def intervals(arm):
    return [tuple(map(exact,pair)) for pair in arm['outer']['retained_intervals']]


def subset(inner,outer):
    return all(any(c<=a<=b<=d for c,d in outer) for a,b in inner)


def widths(arm):
    union=intervals(arm)
    hull=arm['outer']['hull']
    return sum((b-a for a,b in union),Q(0)),None if hull is None else exact(hull[1])-exact(hull[0])


def analyze(output=OUTPUT):
    output=Path(output).resolve()
    if not output.is_relative_to(ROOT/'outputs'):
        raise RuntimeError('exact A4 output required')
    receipt=load(output/'verification/VERIFY.json')
    if (receipt.get('status')!='PASS' or receipt.get('output')!=str(output)
        or load(output/'STAGE_STATUS.json').get('status')!='COMPLETE'
        or load(output/'RUN_STATUS.json').get('status')!='COMPLETE'):
        raise RuntimeError('independent exact-target verification required')
    rows=load(output/'results.json')
    references={(r['task_id'],r['method']):r for r in load(output/'reference/results.json') if r['method'] in METHODS}
    paired=defaultdict(list)
    for row in rows:
        old=references[row['task_id'],row['method']]
        for name in ('depth_mm','initial_ae_mm','case'):
            if row[name]!=old[name]:
                raise RuntimeError('changed paired population')
        group='correct' if exact(row['initial_ae_mm'])==0 else 'negative' if row['case']=='equal_full_negative' else 'informative'
        paired[row['method'],group,''].append((old,row))
        if group!='correct':
            paired[row['method'],'all_shifted',''].append((old,row))
        if group=='informative':
            paired[row['method'],'depth',row['depth_mm']].append((old,row))
        if group!='correct':
            paired[row['method'],'case',row['case']].append((old,row))
    cells=[]
    for (method,group,value),records in sorted(paired.items()):
        n=len(records)
        old_mse=sum((exact(a['se_mm2']) for a,b in records),Q(0))/n
        new_mse=sum((exact(b['se_mm2']) for a,b in records),Q(0))/n
        cells.append({'method':method,'group':group,'value':value,'n':n,
             'old_mse_mm2':str(old_mse),'new_mse_mm2':str(new_mse),'new_minus_old_mse_mm2':str(new_mse-old_mse),
             'better_than_old':sum(exact(b['se_mm2'])<exact(a['se_mm2']) for a,b in records),
             'same_as_old':sum(exact(b['se_mm2'])==exact(a['se_mm2']) for a,b in records),
             'worse_than_old':sum(exact(b['se_mm2'])>exact(a['se_mm2']) for a,b in records),
             'old_repairs':sum(exact(a['actual_gain_mm2'])>0 for a,b in records),
             'new_repairs':sum(exact(b['actual_gain_mm2'])>0 for a,b in records),
             'new_damage_vs_initial':sum(exact(b['actual_gain_mm2'])<0 for a,b in records),
             'old_move_new_keep':sum(a['status']=='MOVE' and b['status']=='KEEP' for a,b in records),
             'old_keep_new_move':sum(a['status']=='KEEP' and b['status']=='MOVE' for a,b in records)})
    new,old=load(output/'outers.json'),load(FULL/'outers.json')
    interval_rows=[]
    for oid,arms in new.items():
        for method in METHODS:
            a,b=old[oid][method],arms[method]
            au,ah=widths(a);bu,bh=widths(b)
            ai,bi=intervals(a),intervals(b)
            ahull,bhull=a['outer']['hull'],b['outer']['hull']
            interval_rows.append({'observation_id':oid,'method':method,
                'union_equal':ai==bi,'global_subset_per_view':subset(ai,bi),
                'hull_equal':ahull==bhull,'global_union_width_mm':str(au),'per_view_union_width_mm':str(bu),
                'global_hull_width_mm':None if ah is None else str(ah),
                'per_view_hull_width_mm':None if bh is None else str(bh),
                'union_width_change_mm':str(bu-au),'hull_width_change_mm':None if ah is None or bh is None else str(bh-ah),
                'left_endpoint_change_mm':None if ahull is None or bhull is None else str(exact(bhull[0])-exact(ahull[0])),
                'right_endpoint_change_mm':None if ahull is None or bhull is None else str(exact(bhull[1])-exact(ahull[1]))})
    interval_summary=[]
    for method in METHODS:
        selected=[row for row in interval_rows if row['method']==method]
        interval_summary.append({'method':method,'n':len(selected),
            'union_equal':sum(row['union_equal'] for row in selected),
            'hull_equal':sum(row['hull_equal'] for row in selected),
            'global_subset_per_view':sum(row['global_subset_per_view'] for row in selected),
            'mean_union_width_change_mm':str(sum((exact(row['union_width_change_mm']) for row in selected),Q(0))/len(selected)),
            'mean_hull_width_change_mm':str(sum((exact(row['hull_width_change_mm']) for row in selected),Q(0))/len(selected))})
    return {'paired':cells,'intervals':interval_rows,'interval_summary':interval_summary,
            'finite_outer_checks':receipt['finite_outer_subset_checks'],
            'source_files_verified':receipt['source_files_verified'],'input_files_verified':receipt['input_files_verified'],
            'scope':'frozen B80 controlled family, hybrid finite-outer algorithm; no physical necessity or external transfer claim'}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--save',action='store_true')
    args=parser.parse_args()
    result=analyze(args.output)
    if args.save:
        destination=Path(args.output).resolve()/'analysis/PAIRED.json'
        destination.parent.mkdir(exist_ok=True)
        with destination.open('x',encoding='utf-8') as stream:
            json.dump(result,stream,indent=2,ensure_ascii=False,allow_nan=False)
            stream.write('\n')
    print(json.dumps(result,ensure_ascii=False))
