"""Check completed scientific outputs and optionally seal the final report bundle."""
from common import *
import argparse, subprocess

def hashes(mapping,base=None):
    for name,digest in mapping.items():
        path=base/name if base is not None else Path(name)
        assert sha(path)==digest,str(path)
    return len(mapping)

def main():
    check_host();parser=argparse.ArgumentParser();parser.add_argument('--seal',action='store_true')
    args=parser.parse_args()
    stages=[ROOT/'evidence'/f'scan{s}' for s in (24,37,55,65,69)]
    stages += [ROOT/'training']+[ROOT/'inference'/f'scan{s}' for s in SCENES]+[ROOT/'evaluation',ROOT/'figures']
    for stage in stages:
        manifest=verify_seal(stage);hashes(manifest.get('source',{}))
    previous={}
    for base,name in ((CROSS,'FINAL_SEAL_V2.json'),(RESERVED,'FINAL_SEAL.json'),(BASE,'FINAL_SEAL.json')):
        previous[str(base)]=hashes(json.loads((base/name).read_text())['files'],base)
    for name in ('AUDIT.json','evaluation/REPRODUCTION.json','figures/VALIDATION.json'):
        assert json.loads((ROOT/name).read_text())['status']=='PASS',name
    summary=json.loads((ROOT/'evaluation/SUMMARY.json').read_text())
    assert summary['rows']==1980 and len(summary['methods'])==33
    assert len(list((ROOT/'inference').rglob('*.ply')))==120
    if args.seal:
        assert not (ROOT/'MANIFEST.json').exists()
        result=subprocess.run([sys.executable,'-B','-m','unittest','-v','test_router.py',
                               'test_support.py','test_evaluation.py'],cwd=ROOT,
                              text=True,capture_output=True)
        assert result.returncode==0,result.stdout+result.stderr
        save_json(ROOT/'VERIFICATION.json',dict(status='PASS',prior_files_unchanged=previous,
            tests=result.stdout+result.stderr,stage_count=len(stages),metric_rows=1980,
            new_pointclouds=120,scope='locked exposed replay; no deployment change'))
        save_json(ROOT/'MANIFEST.json',dict(status='COMPLETE',files={str(p.relative_to(ROOT)):sha(p)
            for p in sorted(ROOT.rglob('*')) if p.is_file()}))
    seal_data=json.loads((ROOT/'MANIFEST.json').read_text())
    count=hashes(seal_data['files'],ROOT)
    actual={str(p.relative_to(ROOT)) for p in ROOT.rglob('*') if p.is_file() and p!=ROOT/'MANIFEST.json'}
    assert actual==set(seal_data['files'])
    print(json.dumps(dict(status='PASS',files=count,previous_unchanged=previous)))

if __name__=='__main__':main()
