"""Seal the completed execution reports and existing result seals."""
import json
from pathlib import Path
from datetime import datetime,timezone
from scene_adapter import sha,save
ROOT=Path(__file__).resolve().parent
DATA=Path('/srv/slam-research/grf/map-denoise/datasets/closeout-confirmation-v1')
paths=[p for p in ROOT.iterdir() if p.is_file() and p.suffix in ('.py','.md','.json')]
paths += [ROOT/('adaptation' if s==40 else 'confirmation')/f'scan{s}'/'SEALED.json' for s in (40,55,65,69)]
paths += [ROOT/'evaluation/SEALED.json']
for folder in [ROOT/'adaptation/scan40',*[ROOT/'confirmation'/f'scan{s}' for s in (55,65,69)],ROOT/'evaluation']:
    for name,h in json.loads((folder/'SEALED.json').read_text())['files'].items():
        if sha(folder/name)!=h:raise RuntimeError('sealed result changed')
for name,h in json.loads((ROOT/'PREPARATION_AUDIT.json').read_text())['sha256'].items():
    if sha(ROOT/name)!=h:raise RuntimeError('preparation changed')
save(ROOT/'EXECUTION_SEAL.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
    host='liekkas',cases=80,ply_outputs=560,metric_rows=560,confirmation_rows=420,
    files={str(p.relative_to(ROOT)):sha(p) for p in paths},
    data_manifests={str(DATA/n):sha(DATA/n) for n in ('INPUT_MANIFEST.json','REFERENCE_MANIFEST.json')},
    official_baseline_run=False,old_preparation_unchanged=True))
print('EXECUTION_SEALED 80 cases / 560 PLY / 560 metric rows')
