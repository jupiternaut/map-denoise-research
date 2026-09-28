"""Read-only verification followed by a new final provenance artifact."""
from common import *


def main():
    check_host()
    for folder in (ROOT/'data', ROOT/'training', ROOT/'evaluation',
                   *(ROOT/'inference'/f'scan{s}' for s in SCENES)):
        verify_seal(folder)
    figures = json.loads((ROOT/'figures/manifest.json').read_text())
    for path, expected in figures['sources_sha256'].items():
        if sha(path) != expected:
            raise AssertionError('figure source changed')
    for name, expected in figures['outputs_sha256'].items():
        if sha(ROOT/'figures'/name) != expected:
            raise AssertionError('figure artifact changed')
    for name in ('REPORT.md','CHECKPOINT.md','COMMANDS.md','AUDIT_REPORT.md'):
        if not (ROOT/name).is_file():
            raise AssertionError('missing delivery')
    files = {str(p.relative_to(ROOT)):sha(p) for p in sorted(ROOT.rglob('*'))
             if p.is_file() and p.name != 'FINAL_SEAL.json'}
    save_json(ROOT/'FINAL_SEAL.json', dict(status='COMPLETE_EXPOSED_REPLAY',
              files=files, independent_test_scenes=False, old_sources_read_only=True))
    print('FINAL SEALED', len(files), 'files', flush=True)


if __name__ == '__main__':
    main()
