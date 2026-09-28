"""Check immutable sources and artifacts without rerunning experiments."""
from common import *
import argparse,subprocess

def check_map(files):
    for name,digest in files.items():
        if sha(Path(name))!=digest:raise AssertionError('changed: '+name)

def main():
    check_host()
    ap=argparse.ArgumentParser();ap.add_argument('--seal',action='store_true');args=ap.parse_args()
    folders=[OUT/'observation_smoke'/'scan55']
    folders += [OUT/'inference'/f'scan{s}' for s in SCENES]
    folders += [OUT/'evaluation'/f'scan{s}' for s in SCENES]
    folders += [OUT/'evaluation',OUT/'figures']
    checked=0
    for folder in folders:
        record=verify_seal(folder)
        check_map(record.get('source',{}));checked+=len(record.get('source',{}))
    old=json.loads((VIS_CODE/'MANIFEST.json').read_text())['files']
    check_map(old)
    if args.seal:
        if (ROOT/'MANIFEST.json').exists():raise AssertionError('already sealed')
        tests=subprocess.run([sys.executable,'-B','-m','unittest','-v','test_segment_oracle.py',
                              'test_step_observation.py','test_evaluation.py'],cwd=ROOT,text=True,capture_output=True)
        if tests.returncode:raise AssertionError(tests.stdout+tests.stderr)
        for p in (ROOT/'AUDIT.json',OUT/'evaluation/REPRODUCTION.json',OUT/'figures/VALIDATION.json'):
            if json.loads(p.read_text())['status']!='PASS':raise AssertionError(str(p))
        save_json(ROOT/'VERIFICATION.json',dict(status='PASS',tests=tests.stdout+tests.stderr,
            previous_visibility_files_unchanged=len(old),stage_count=len(folders),source_hash_checks=checked))
        files={str(p):sha(p) for base in (ROOT,OUT) for p in sorted(base.rglob('*')) if p.is_file()}
        save_json(ROOT/'MANIFEST.json',dict(status='COMPLETE',files=files))
    final=json.loads((ROOT/'MANIFEST.json').read_text())
    check_map(final['files'])
    actual={str(p) for base in (ROOT,OUT) for p in base.rglob('*') if p.is_file() and p!=ROOT/'MANIFEST.json'}
    if actual!=set(final['files']):raise AssertionError('final inventory changed')
    print(json.dumps(dict(status='PASS',files=len(actual),previous_visibility_unchanged=len(old))))

if __name__=='__main__':main()
