"""Post-lock algebraic diagnostic; does not change a single prediction.

Decompose full-patch NCC into center-within, ring-within, between-group terms.
No truth file or candidate data is read. Depth is the already observed support mean.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[k] = '1'
import json
from pathlib import Path
import numpy as np
from PIL import Image
from support import sample, camera, rays, project

ROOT = Path(__file__).resolve().parent


def mean_union(parts):
    length = sum(b-a for a,b in parts)
    return sum((b-a)*(a+b)/2 for a,b in parts)/length if length else sum(a for a,b in parts)/len(parts)


def main():
    requests = json.loads((ROOT/'REQUESTS.json').read_text())
    observed = json.loads((ROOT/'OBSERVATIONS.json').read_text())['rows']
    images,cams = {},{}
    for sn,scene in requests['scenes'].items():
        for name,c in scene['cameras'].items():
            with Image.open(c['image_path']) as im:
                images[sn,name] = np.asarray(im.convert('L').resize((c['width'],c['height']),Image.Resampling.BILINEAR),float)
            cams[sn,name] = camera(c)
    yy,xx = np.mgrid[-4:5,-4:5]
    offsets = np.column_stack((xx.ravel(),yy.ravel()))
    center = ((abs(xx)<=1)&(abs(yy)<=1)).ravel()
    n1,n2 = int(center.sum()),int((~center).sum())
    rows = []
    for row in observed:
        ev = row['evidence']['translation_full9']
        if not ev['intervals']:
            continue
        sn=str(row['scene']); scene=requests['scenes'][sn]; ref=scene['reference']
        z=mean_union(ev['intervals'])
        a,_ = sample(images[sn,ref],np.asarray(row['pixel_xy'])+offsets)
        target=cams[sn,ref]['C']+z*rays(cams[sn,ref],np.asarray(row['pixel_xy']))
        for src in scene['sources']:
            uv,_=project(cams[sn,src],target)
            b,valid=sample(images[sn,src],uv+offsets)
            ac=a-a.mean();bc=b-b.mean();den=np.linalg.norm(ac)*np.linalg.norm(bc)
            within_center=float(np.dot(a[center]-a[center].mean(),b[center]-b[center].mean()))
            within_ring=float(np.dot(a[~center]-a[~center].mean(),b[~center]-b[~center].mean()))
            between=float(n1*n2/(n1+n2)*(a[center].mean()-a[~center].mean())*(b[center].mean()-b[~center].mean()))
            total=float(np.dot(ac,bc))
            if not np.isclose(total,within_center+within_ring+between,atol=1e-9,rtol=1e-12):
                raise RuntimeError('covariance identity failed')
            denom_a=float(np.dot(ac,ac))
            rows.append(dict(scene=row['scene'],roi=row['roi'],query=row['query'],source=src,
                target_support_mean_mm=z,mask_count=row['evidence']['translation_connected9']['scans'][0]['mask_count'],
                center_std=float(a[center].std()),full_std=float(a.std()),
                center_within_energy_fraction=float(np.sum((a[center]-a[center].mean())**2)/denom_a),
                ncc=total/den,center_ncc_contribution=within_center/den,ring_ncc_contribution=within_ring/den,
                between_ncc_contribution=between/den,valid=bool(valid.all())))
    out=dict(scope='post-hoc algebraic image diagnostic, no policy tuning or physical labels',
        formula='NCC=(within-center covariance + within-ring covariance + between-group covariance)/full centered norms',
        note='support mean can lie between components; scores here need not pass the threshold',rows=rows)
    with (ROOT/'evaluation/NCC_DECOMPOSITION.json').open('x') as f:
        json.dump(out,f,indent=2,allow_nan=False);f.write('\n')
    for r in rows:
        print(r['roi'],r['query'],r['source'],'center_std',round(r['center_std'],3),
              'center_energy',round(r['center_within_energy_fraction'],5),
              'ncc_terms',*[round(r[k],4) for k in ('center_ncc_contribution','ring_ncc_contribution','between_ncc_contribution')])


if __name__=='__main__':
    main()
