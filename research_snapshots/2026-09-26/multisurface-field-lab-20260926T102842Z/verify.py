"""Final artifact/source integrity and executable test record, no method tuning."""
import common as c
import csv,json,subprocess,sys,time,platform,importlib
from pathlib import Path


def main():
    c.check_host();start=time.monotonic()
    test=subprocess.run([sys.executable,'-B','-m','unittest','discover','-v'],
        cwd=c.ROOT,capture_output=True,text=True)
    if test.returncode:raise RuntimeError(test.stdout+test.stderr)
    sealed=[];sources={}
    for stage in ('inference','evaluation'):
        for sid in c.SCENES:
            path=c.OUT/stage/f'scan{sid}'
            record=c.verify_seal(path);sealed.append(str(path))
            for name,digest in record['source'].items():
                if name in sources and sources[name]!=digest:raise AssertionError('conflicting source hashes')
                sources[name]=digest
    record=c.verify_seal(c.OUT/'evaluation');sealed.append(str(c.OUT/'evaluation'))
    sources.update(record['source'])
    for name,digest in sources.items():
        if c.sha(name)!=digest:raise AssertionError('source changed: '+name)
    with (c.OUT/'evaluation/METRICS.csv').open() as stream:rows=list(csv.DictReader(stream))
    if len(rows)!=720:raise AssertionError('evaluation incomplete')
    actual=list((c.OUT/'inference').glob('scan*/scan*/*.ply'))
    if len(actual)!=420:raise AssertionError('expected 60 times 7 actual PLY')
    evidence=list((c.OUT/'inference').glob('scan*/scan*/EVIDENCE.npz'))
    if len(evidence)!=60:raise AssertionError('missing evidence')
    checks=json.loads((c.OUT/'evaluation/REPRODUCTION.json').read_text())['checks']
    reproduction_max=max(x['max_difference'] for x in checks)
    result=dict(status='PASS',host='liekkas',sealed=sealed,verified_source_files=len(sources),
        metric_rows=len(rows),cases=60,actual_PLY=len(actual),case_evidence=len(evidence),
        reproduction_checks=len(checks),reproduction_max_difference=reproduction_max,
        test_returncode=test.returncode,test_log=test.stdout+test.stderr,
        seconds=time.monotonic()-start,reference_used_only_in_evaluation=True,
        old_smoke='retained before duplicate-UV graph repair; excluded from evaluation')
    result['environment']=dict(python=sys.version,platform=platform.platform(),
        libraries={x:importlib.import_module(x).__version__ for x in ('numpy','scipy','open3d','matplotlib','threadpoolctl')})
    c.save_json(c.ROOT/'VERIFICATION.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='test_log'},indent=2))

if __name__=='__main__':main()
