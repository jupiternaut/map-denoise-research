"""Create and verify an exclusive source/results snapshot without touching old work."""
import datetime
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parent
RUN = Path('/srv/slam-research/grf/map-denoise/runs/six-track-v1-20260911-1600')
DEST = Path('/srv/slam-research/grf/map-denoise/checkpoints/20260911-six-track-v1')


def digest_stream(stream):
    h = hashlib.sha256()
    for block in iter(lambda: stream.read(1 << 20), b''):
        h.update(block)
    return h.hexdigest()


def main():
    validation = json.loads((ROOT/'FINAL_VALIDATION.json').read_text())
    assert validation['all_passed']
    DEST.mkdir(parents=True, exist_ok=False)
    files = []
    for path, prefix in [(ROOT, 'workspace'), (RUN, 'results')]:
        for item in sorted(path.rglob('*')):
            if not item.is_file() or '__pycache__' in item.parts:
                continue
            if item.is_symlink():
                raise RuntimeError(f'Unexpected symlink: {item}')
            with item.open('rb') as stream:
                digest = digest_stream(stream)
            files.append({'source': str(item), 'archive_path': f'{prefix}/{item.relative_to(path)}',
                          'bytes': item.stat().st_size, 'sha256': digest})
    manifest = {'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'scope': 'New-round source/reports and all new raw results; external dependencies excluded',
                'files': files}
    with (DEST/'FILES.json').open('x') as output:
        json.dump(manifest, output, ensure_ascii=False, indent=2)
    archive = DEST/'six-track-source-results.tar.gz'
    with tarfile.open(archive, 'x:gz', compresslevel=6) as tar:
        for item in files:
            tar.add(item['source'], arcname=item['archive_path'], recursive=False)
        tar.add(DEST/'FILES.json', arcname='FILES.json')
    failures = []
    with tarfile.open(archive, 'r:gz') as tar:
        assert len(tar.getmembers()) == len(files)+1
        for item in files:
            member = tar.getmember(item['archive_path'])
            with tar.extractfile(member) as stream:
                if digest_stream(stream) != item['sha256']:
                    failures.append(item['archive_path'])
    with archive.open('rb') as stream:
        archive_hash = digest_stream(stream)
    with (DEST/'SHA256SUMS').open('x') as output:
        output.write(f'{archive_hash}  {archive.name}\n')
    report = {'archive': str(archive), 'archive_bytes': archive.stat().st_size,
              'archive_sha256': archive_hash, 'files_verified': len(files),
              'uncompressed_file_bytes': sum(item['bytes'] for item in files),
              'mismatches': failures, 'all_passed': not failures}
    with (DEST/'PACKAGE_VERIFICATION.json').open('x') as output:
        json.dump(report, output, indent=2)
    print(json.dumps(report), flush=True)
    if failures:
        raise RuntimeError('Archive member verification failed')


if __name__ == '__main__':
    main()
