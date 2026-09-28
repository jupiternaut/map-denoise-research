"""Final artifact/source integrity and focused tests; no regeneration."""
from common import *
from learner import ARMS
import argparse,subprocess

def check_files(files,base=None):
    for name,digest in files.items():assert sha(base/name if base else Path(name))==digest,name

def main():
    check_host();ap=argparse.ArgumentParser();ap.add_argument('--seal',action='store_true');args=ap.parse_args()
    folders=[OUT/'evidence'/f'scan{s}' for s in (24,37,55,65,69)]
    folders += [OUT/'training/arms'/a for a in ARMS]+[OUT/'training']
    folders += [OUT/'inference'/f'scan{s}' for s in SCENES]+[OUT/'evaluation',OUT/'figures']
    for folder in folders:
        record=verify_seal(folder);check_files(record.get('source',{}))
    previous=json.loads((JOINT/'MANIFEST.json').read_text())['files'];check_files(previous,JOINT)
    for base,name in ((prior.CROSS,'FINAL_SEAL_V2.json'),(prior.RESERVED,'FINAL_SEAL.json'),(prior.BASE,'FINAL_SEAL.json')):
        check_files(json.loads((base/name).read_text())['files'],base)
    if args.seal:
        assert not (ROOT/'MANIFEST.json').exists()
        test=subprocess.run([sys.executable,'-B','-m','unittest','-v','test_visibility.py','test_learner.py','test_evaluation.py'],
            cwd=ROOT,text=True,capture_output=True);assert test.returncode==0,test.stdout+test.stderr
        for path in (ROOT/'AUDIT.json',OUT/'evaluation/REPRODUCTION.json',OUT/'figures/VALIDATION.json'):
            assert json.loads(path.read_text())['status']=='PASS',str(path)
        save_json(ROOT/'VERIFICATION.json',dict(status='PASS',tests=test.stdout+test.stderr,
            prior_joint_files_unchanged=len(previous),stage_count=len(folders)))
        save_json(ROOT/'MANIFEST.json',dict(status='COMPLETE',files={str(p):sha(p) for base in (ROOT,OUT)
            for p in sorted(base.rglob('*')) if p.is_file()}))
    manifest=json.loads((ROOT/'MANIFEST.json').read_text());check_files(manifest['files'])
    actual={str(p) for base in (ROOT,OUT) for p in base.rglob('*') if p.is_file() and p!=ROOT/'MANIFEST.json'}
    assert actual==set(manifest['files'])
    print(json.dumps(dict(status='PASS',files=len(actual),previous_joint_unchanged=len(previous))))

if __name__=='__main__':main()
