"""Verify original immutable snapshot and its additive wording correction."""
from common import ROOT, PREV, BASE, sha, save_json, check_host
import argparse
import json


def verify(mapping, folder):
    for name, expected in mapping.items():
        assert sha(folder / name) == expected, str(folder / name)
    return len(mapping)


def main():
    check_host()
    parser = argparse.ArgumentParser()
    parser.add_argument('--seal-addendum', action='store_true')
    args = parser.parse_args()
    original = json.loads((ROOT / 'FINAL_SEAL.json').read_text())
    old_count = verify(original['files'], ROOT)
    prior_counts = {}
    for prior in (PREV, BASE):
        manifest = json.loads((prior / 'FINAL_SEAL.json').read_text())
        prior_counts[str(prior)] = verify(manifest['files'], prior)
    path = ROOT / 'FINAL_SEAL_V2.json'
    if args.seal_addendum:
        if path.exists():
            raise FileExistsError(path)
        files = {str(p.relative_to(ROOT)): sha(p) for p in sorted(ROOT.rglob('*'))
                 if p.is_file()}
        added = set(files) - set(original['files']) - {'FINAL_SEAL.json'}
        assert added == {'ERRATA.md', 'verify_release.py'}, added
        save_json(path, dict(status='COMPLETE_EXPOSED_REPLAY_WITH_WORDING_ERRATA',
                             original_files_unchanged=old_count, files=files))
    current = json.loads(path.read_text())
    count = verify(current['files'], ROOT)
    actual = {str(p.relative_to(ROOT)) for p in ROOT.rglob('*')
              if p.is_file() and p != path}
    assert actual == set(current['files']), 'unsealed added/deleted files'
    print(json.dumps(dict(status='PASS', final_files=count,
                          original_files_unchanged=old_count,
                          prior_files_unchanged=prior_counts)))


if __name__ == '__main__':
    main()
