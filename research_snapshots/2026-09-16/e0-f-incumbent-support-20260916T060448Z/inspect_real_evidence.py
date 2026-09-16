"""Read-only artifact inventory, not a real geometry experiment or GT evaluation."""
from pathlib import Path
import numpy as np
from run_experiment import ROOT, sha, write_json


def main():
    base=Path('/home/grf/Documents/Codex/2026-09-15/map-denoise-v25-surface-regeneration')
    source=base/'src/v25/evidence.py'
    rows=[]
    for roi in ('scan24_turret_join','scan37_scissor_cross','scan24_wall_control','scan37_stone_control'):
        path=base/'runs/2026-09-14T165314Z'/roi/'evidence.npz'
        before=sha(path)
        with np.load(path,allow_pickle=False) as z:
            # Skip object metadata deliberately; do not unpickle to inspect arrays.
            score=z['zncc'];count=z['n_valid_src'];depth=z['depths_norm'];names=z['view_names']
            rows.append(dict(roi=roi,path=str(path),sha256=before,keys=z.files,
                view_names=names.tolist(),ref_name=str(z['ref_name']),shape=list(score.shape),
                finite_fraction=float(np.isfinite(score).mean()),
                valid_source_count_range=[int(count.min()),int(count.max())],
                depth_step_normalized=float(np.median(np.diff(depth))),
                per_view_costs_saved=False,contains_evaluator_truth=False,
                note='zncc is arithmetic source-view mean; counts are valid warps/windows, not occlusion-certified visibility'))
        assert sha(path)==before
    write_json(ROOT/'REAL_EVIDENCE_INVENTORY.json',dict(rows=rows,
        source=str(source),source_sha256=sha(source),
        source_mechanism='build_evidence accumulates source scores then acc/n_ok; save_bundle omits source-view axis',
        required_for_next_test='recompute per-source scores from original images/cameras in a new directory; add explicit visibility handling',
        scope='read-only inventory only, no real performance or layer-discrimination claim'))
    print('Inventoried',len(rows),'real bundles; none contain per-view cost volumes.')


if __name__=='__main__':main()
