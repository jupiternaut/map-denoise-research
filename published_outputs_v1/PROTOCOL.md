# Published reconstruction output pilot

Input: original 3DGS room iteration 30000; GeoSVR complete scan24 and Courthouse meshes.
No retraining, synthetic noise injection, scan ID fabrication, or evaluator GT input.
Old checkpoints and downloaded originals are read-only to the experiment.

1. Validate full file schemas, finite XYZ, face indices, Gaussian parameters; record hashes and sizes.
2. Select three deterministic 1024-nearest-vertex neighborhoods per scene, around
   vertices nearest the coordinate-wise 25%, 50%, and 75% scene quantiles. No manual selection by success.
3. Reuse the existing spatial mixture solver ONLY as a geometry-only adapter: estimate
   patch PCA basis, normalize by measured median nearest-neighbor spacing, no frame correction.
   This is NOT the previously validated source-aware algorithm. Test residual scales 0.5, 1, 2
   times spacing; these are sensitivity settings, not known sensor noise or metric millimeters.
4. Compare identity, single-plane projection, existing spatial mixture (36 iterations),
   and official APSS scale 4 on identical measured patches. Keep failures visible.
5. Report displacement, neighbor edge distortion and incident mesh triangle changes, including
   triangles crossing the edited patch boundary. Save patch outputs and before/after geometric plots.
   Gaussian centers are NOT assumed to be surface samples; their projection is a diagnostic only.
6. For room additionally export a full Gaussian model using an explicitly named baseline:
   remove only sigmoid(opacity) < 0.01, preserving all other attributes exactly.
   This is not our novel operator, a visibility proof, or verified improvement in rendering.
7. No independent reference geometry or evaluation photographs are in this download budget.
   Therefore the outcome is input/transfer feasibility and deformation diagnostics, NOT a
   geometry accuracy benchmark. Reduced roughness, fewer points, and smaller file size do not prove denoising.

Scope: full input audit on all three scenes; local geometry tests on 3 x 1024 points per scene;
one full-scene Gaussian pruning baseline. No full-scene mesh smoothing is claimed.
