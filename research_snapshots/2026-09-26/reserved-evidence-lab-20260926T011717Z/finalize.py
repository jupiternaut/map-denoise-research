"""Verify the completed queue and seal it without changing scientific outputs."""
from common import *
import argparse, subprocess, time


def check_hashes(mapping, base=None):
    for name, digest in mapping.items():
        p = (base / name) if base is not None else Path(name)
        if sha(p) != digest:
            raise AssertionError(f'hash mismatch: {p}')
    return len(mapping)


def main():
    check_host()
    ap=argparse.ArgumentParser()
    ap.add_argument('--verify-only',action='store_true')
    args=ap.parse_args()
    start=time.monotonic()
    stages=[ROOT/'evidence'/f'scan{s}' for s in (24,37,55,65,69)]
    stages += [ROOT/'training']+[ROOT/'inference'/f'scan{s}' for s in SCENES]+[ROOT/'evaluation']
    counts={}
    for folder in stages:
        manifest=verify_seal(folder)
        check_hashes(manifest.get('source',{}))
        counts[str(folder.relative_to(ROOT))]=len(manifest['files'])
    previous=json.loads((PREV/'FINAL_SEAL.json').read_text())
    prior_count=check_hashes(previous['files'],PREV)
    figure=json.loads((ROOT/'figures/MANIFEST.json').read_text())
    check_hashes(figure['source_sha256'])
    check_hashes(figure['output_sha256'],ROOT/'figures')
    audits={}
    for name in ('AUDIT_EVIDENCE.json','AUDIT_SELECTION.json','AUDIT_METRICS.json',
                 'figures/VALIDATION.json','evaluation/REPRODUCTION.json'):
        status=json.loads((ROOT/name).read_text())['status']
        if status!='PASS':raise AssertionError((name,status))
        audits[name]=status
    summary=json.loads((ROOT/'evaluation/SUMMARY.json').read_text())
    assert summary['rows']==2460 and len(summary['methods'])==41
    assert len(list((ROOT/'inference').rglob('*.ply')))==240
    if args.verify_only:
        final=json.loads((ROOT/'FINAL_SEAL.json').read_text())
        count=check_hashes(final['files'],ROOT)
        print(json.dumps(dict(status='PASS',final_files=count,prior_files=prior_count,
                              stage_files=counts,seconds=time.monotonic()-start)))
        return
    for name in ('FINAL_VERIFICATION.json','FINAL_SEAL.json'):
        if (ROOT/name).exists():raise FileExistsError(ROOT/name)
    test=subprocess.run([sys.executable,'-B','-m','unittest','-v','test_paired_features.py'],
                         cwd=ROOT,text=True,capture_output=True,check=False)
    if test.returncode:raise AssertionError(test.stdout+test.stderr)
    save_json(ROOT/'FINAL_VERIFICATION.json',dict(status='PASS',host='liekkas',
        stage_files=counts,prior_final_seal_files_unchanged=prior_count,audits=audits,
        tests=dict(command=test.args,returncode=test.returncode,output=test.stdout+test.stderr),
        evaluation_rows=2460,exported_ply=240,wall_seconds=time.monotonic()-start,
        scope='EXPOSED_REPLAY_NOT_GLOBAL_ARCHITECTURE_OPTIMALITY'))
    files={str(p.relative_to(ROOT)):sha(p) for p in sorted(ROOT.rglob('*')) if p.is_file()}
    save_json(ROOT/'FINAL_SEAL.json',dict(status='COMPLETE_EXPOSED_REPLAY',files=files))
    print(json.dumps(dict(status='PASS',final_files=len(files),prior_files=prior_count)))


if __name__=='__main__':main()
