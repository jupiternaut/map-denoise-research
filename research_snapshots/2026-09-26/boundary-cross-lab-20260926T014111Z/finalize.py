"""Verify completed experiment and freeze all files; no scientific outputs changed."""
from common import *
import argparse
import csv
import subprocess
import time


def check_hashes(mapping, base=None):
    for name, digest in mapping.items():
        path = base / name if base is not None else Path(name)
        if sha(path) != digest:
            raise AssertionError(f'hash mismatch: {path}')
    return len(mapping)


def main():
    check_host()
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    start = time.monotonic()
    stages = [ROOT / 'data']
    stages += [ROOT / 'evidence' / f'scan{s}' for s in (24, 37, 55, 65, 69)]
    stages += [ROOT / 'training']
    stages += [ROOT / 'inference' / f'scan{s}' for s in SCENES]
    stages += [ROOT / 'evaluation']
    counts = {}
    for folder in stages:
        manifest = verify_seal(folder)
        check_hashes(manifest.get('source', {}))
        counts[str(folder.relative_to(ROOT))] = len(manifest['files'])
    previous = {}
    for prior in (PREV, BASE):
        manifest = json.loads((prior / 'FINAL_SEAL.json').read_text())
        previous[str(prior)] = check_hashes(manifest['files'], prior)
    figure = json.loads((ROOT / 'figures/MANIFEST.json').read_text())
    check_hashes(figure['source_sha256'])
    check_hashes(figure['output_sha256'], ROOT / 'figures')
    audit_hashes = json.loads((ROOT / 'AUDIT_HASHES.json').read_text())
    check_hashes(audit_hashes['sha256'], ROOT)
    audit_files = ('AUDIT_PROVENANCE.json', 'AUDIT_DATA.json', 'AUDIT_EVIDENCE.json',
                   'AUDIT_SELECTION.json', 'AUDIT_METRICS.json', 'AUDIT_HASHES.json',
                   'data/B_LABEL_REPRODUCTION.json', 'evaluation/REPRODUCTION.json',
                   'figures/VALIDATION.json', 'figures/VISUAL_REVIEW.json')
    audits = {}
    for name in audit_files:
        payload = json.loads((ROOT / name).read_text())
        if name == 'AUDIT_PROVENANCE.json':
            # This frozen audit uses named assertions rather than a status field.
            assert payload['host'] == 'liekkas'
            assert payload['archived_inputs_modified'] is False
            assert payload['replay_gt_read'] is False
            assert payload['sealed_files_verified'] == 468
            assert payload['cases_verified'] == 84
            assert payload['schema_width_A'] == payload['schema_width_B'] == 64
            for key in ('schema_names_equal', 'first_eight_input_features_equal',
                        'original_view_order_matches_prior_reserved_lab',
                        'geometry_and_features_finite', 'same_point_rows',
                        'development_state_input_equals_identity',
                        'replay_native_mesh_row_ids_unique_and_length_match',
                        'previous_pair_row_ids_valid', 'replay_pair_row_ids_equal_arange'):
                assert payload[key] is True, key
            status = 'PASS'
        else:
            status = payload['status']
        if status != 'PASS':
            raise AssertionError((name, status))
        audits[name] = status
    summary = json.loads((ROOT / 'evaluation/SUMMARY.json').read_text())
    assert summary['rows'] == 2640 and len(summary['methods']) == 44
    with (ROOT / 'evaluation/METRICS.csv').open(newline='') as stream:
        assert sum(1 for _ in csv.DictReader(stream)) == 2640
    assert len(list((ROOT / 'inference').rglob('*.ply'))) == 180
    if args.verify_only:
        final = json.loads((ROOT / 'FINAL_SEAL.json').read_text())
        count = check_hashes(final['files'], ROOT)
        actual = {str(p.relative_to(ROOT)) for p in ROOT.rglob('*')
                  if p.is_file() and p.name != 'FINAL_SEAL.json'}
        assert actual == set(final['files']), 'unsealed added/deleted files'
        print(json.dumps(dict(status='PASS', final_files=count,
                              prior_files=previous, stage_files=counts,
                              seconds=time.monotonic() - start)))
        return
    for name in ('FINAL_VERIFICATION.json', 'FINAL_SEAL.json'):
        if (ROOT / name).exists():
            raise FileExistsError(ROOT / name)
    result = subprocess.run([sys.executable, '-B', '-m', 'unittest', '-v',
                             'test_candidate_contract.py'], cwd=ROOT, text=True,
                            capture_output=True, check=False)
    if result.returncode:
        raise AssertionError(result.stdout + result.stderr)
    save_json(ROOT / 'FINAL_VERIFICATION.json', dict(
        status='PASS', host='liekkas', stage_files=counts,
        prior_final_seal_files_unchanged=previous, audits=audits,
        tests=dict(command=result.args, returncode=result.returncode,
                   output=result.stdout + result.stderr),
        evaluation_rows=2640, evaluation_arms=44, replay_cases=60, exported_ply=180,
        wall_seconds=time.monotonic() - start,
        scope='EXPOSED_REPLAY_CANDIDATE_EVIDENCE_CROSS_NOT_GLOBAL_OPTIMALITY'))
    files = {str(p.relative_to(ROOT)): sha(p) for p in sorted(ROOT.rglob('*'))
             if p.is_file()}
    save_json(ROOT / 'FINAL_SEAL.json', dict(status='COMPLETE_EXPOSED_REPLAY', files=files))
    print(json.dumps(dict(status='PASS', final_files=len(files), prior_files=previous)))


if __name__ == '__main__':
    main()
