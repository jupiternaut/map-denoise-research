"""Read-only V5 result aggregation, optionally save a non-overwriting JSON."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'exploration_v4'))
import summarize_v4
summarize_v4.PAIRS=(('pool_compatible','shared_group_slope'),('pool_compatible','node_intercepts'),
                   ('pool_independent','shared_group_slope'),('pool_independent','node_intercepts'),
                   ('identity','official_gicp'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('run',type=Path);parser.add_argument('--save',action='store_true')
    args=parser.parse_args();result=summarize_v4.summarize(args.run)
    if args.save:
        with (args.run/'AGGREGATES.json').open('x') as f:json.dump(result,f,indent=2)
    main=result['groups']['full_identifiable']
    print('Full identifiable inputs',main['inputs'])
    for method,values in main['methods'].items():
        print(method,{key:round(values[key]['mean'],6) if values[key] else None for key in
                      ('surface_accuracy_mean_mm','matched_point_rms_mm','fitted_gap_at_same_xy_error_mm','method_seconds')})
    for pair,values in main['paired'].items():
        print(pair,{key:{k:val for k,val in values[key].items() if k!='cases'} for key in
                    ('surface_accuracy_mean_mm','fitted_gap_at_same_xy_error_mm') if key in values})
