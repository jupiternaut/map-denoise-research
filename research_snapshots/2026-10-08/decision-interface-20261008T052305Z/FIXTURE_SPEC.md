# Frozen synthetic fixture specification

Registered before dataset generation. Target host: liekkas. Producer file: `new_fixtures.py`. This specification implements the approved decision-interface plan without changing the sealed image estimator or reading old per-object outcomes.

## Stages and object allocation

`create_dataset(root, stage)` accepts stage `calibration` or `confirmation`. Each call creates48 independent objects: four mechanisms (`flat_contrast`, `flat_equal`, `textured_boundary`, `textured_single`) × three batches × four objects. Calibration batch seeds are31001/31002/31003; confirmation batch seeds are41001/41002/41003. NumPy PCG64 streams use SeedSequence([batch_seed, mechanism_index, within_batch_index]); these namespaces distinguish object draws. A separate stream SeedSequence([all three stage batch seeds,991]) permutes all48 rows before opaque `o000`–`o047` IDs are assigned. IDs and observed row order do not encode mechanism, geometry, seed, or grid status.

For each mechanism and batch, within-batch indices0/1 use a depth sampled uniformly from public action-grid values525:15:690. Indices2/3 use a continuous uniform depth in[520,690], redrawing only if its distance to the public450:15:900 grid is<=1e-6mm. Thus each mechanism has exactly six on-grid and six off-grid objects; no world is fixed at600. Independent draws may share an on-grid depth. There are no background-pair replicas or three-incumbent replicas at image-generation time. The runner later creates the correct and±60mm task states without passing their labels to inference.

Generation is explicit only: importing the module does not create files. This producer must not be called for confirmation until the main runner records the registered B2 gate as passed. The main runner owns that gate; the producer does not infer authorization from existing files or generate B3 automatically. Tests do not create either48-object stage.

## Geometry, camera, appearance

Images are grayscale float64 arrays of shape(3,128,128), units0–255, with no added sensor noise, exposure perturbation, quantization, or clipping. Supplied cameras are exact. Intrinsics are f_x=f_y=160, principal point(64,64). Reference R=I,C=(0,0,0). Source order is sign−1 then+1: C=(sign×60,0,0), world-to-camera R=Ry(sign×0.07) @ Rz(sign×0.11). Source K is the same as reference K.

The two boundary mechanisms use an opaque slanted rectangle over a full opaque background plane. True foreground depth is reference optical Z. Rectangle center is(64,64)+independent U[−0.5,0.5] offsets; `half_width` is U[1.2,3.2] reference pixels, with the same half-width interpretation as the prior fixture (full width2.4–6.4). Half-height remains8 reference pixels and rotation remains0.17rad. The rectangle is constructed in reference coordinates and lies on the foreground plane. The background depth is true_depth+U[150,300]mm. Width, phase, gap, radiance, and texture draws are independently sampled within each object's stream. Shape parameters for single-plane and equal-color controls are null, so a nonexistent boundary is not varied to create nominal sample diversity.

For both boundary mechanisms, background level is U[25,65], and positive foreground-minus-background level contrast is independently U[85,140]. The foreground remains brighter than the background even after bounded textures: no contrast sign or hard-case selection is made from results. This sign convention matches the locked estimator's intended initialization regime and is part of the tested domain.

`flat_contrast` uses those constant foreground/background radiances. `textured_boundary` adds independent continuous textures on each plane. A texture is the mean of four cosine components, multiplied by its amplitude. Each component has wavelength U[4,12] reference pixels, direction U[−pi,pi], and phase U[−pi,pi]. The foreground amplitude is U[8,18]; background amplitude is U[5,12]. The functions use fixed reference-coordinate arguments (u−64,v−64) and no view-dependent radiance. These arrays and phases are producer-only metadata. The choice of four bounded components and these ranges is fixed before data and is not selected by prediction performance.

`textured_single` is a full opaque plane at true_depth, with level U[70,190] and the same independently drawn four-component texture family, amplitude U[15,30]. It has no competing background or target boundary.

`flat_equal` is exactly one scalar everywhere in all three images. To ensure12 distinct observed objects without fake width replicas, the gray interval[20,235] is split into12 nonoverlapping equal bins indexed by(batch_index×4+within_batch_index); each object draws its scalar uniformly inside its bin. Each object has a separately sampled evaluation depth, but its images are mathematically independent of that depth and camera. The implementation directly fills the constant image array, avoiding summation-rounding differences between views. No hidden silhouette, texture, noise, or boundary metadata enters inference.

## Independent image integration

The producer imports neither predictor.py nor old fixture code. The only reused facts are the documented physical camera convention and scene ranges. It implements a plane homography from reference pixels to each source image:

H(z)=K_v [R_v R_ref^T + (R_v(C_ref−C_v)) e3^T/z] K_ref^−1.

For every source pixel,49 midpoint positions with offsets((i+0.5)/7−0.5), i=0..6, are independently mapped back through H(z)^−1 for each plane. Foreground membership is evaluated in the rectangle's local rotated axes, without predictor polygon routines. The applicable continuous foreground or background radiance is averaged over49 samples. Processing uses16 sensor rows per tile by default; this bounds temporary memory without changing integration. The primary producer always uses7×7; optional sample-count arguments exist only for analytical implementation tests, not data selection.

The integration code is separate from the frozen nine-ray predictor and uses a different coordinate-mapping implementation. It is a deliberately controlled renderer mismatch, not a claim of independent real-world data or a wholly different appearance family.

## Data boundary, files, reproducibility

`root/observed/inputs.json` is a JSON array with only{id,cameras,image_file,sha256}. Each `image_file` is a root-relative `observed/oNNN.npz` containing only an `images` array. No mode, mechanism, true depth, source seed, producer name, grid flag, or phase is saved in observed files. Camera records contain onlyK,R,C. SHA256 refers to the exact NPZ file bytes.

`root/truth/metadata.json` is a JSON array with{id,true_depth,mechanism,producer_parameters}. Producer parameters retain stage, batch seed/index, within-batch index, on-grid status, geometry, radiance, complete texture parameters, and the known true depth for reproducibility. Prediction processes receive the observed manifest only. Calibration geometry is opened separately by the runner for its budgeted calibration endpoint; confirmation truth is opened only after prediction sealing.

The function refuses to overwrite either an existing observed or truth directory. Files are created exclusively. It returns paths and counts; it never evaluates a model. There is no dataset-generating command executed by the unit tests. The runner can time exactly the first two selected calibration objects by calling `make_object(stage,batch_index,mechanism,within_batch_index)` then `render_object(spec)`; this uses the same producer and objects as the eventual dataset and need not open confirmation.

## Validation scope

Handcrafted tests check: camera matrices; homography versus an independent direct 3D projection; exact constant controls; pixel midpoint mixing and an independent hand-counted stripe; exact integration of a linear texture on a single plane; view displacement; tiling invariance; array dtype/shape; and absence of predictor/old-fixture imports. These establish geometry and numerical contracts, not scientific success. Full calibration and conditional confirmation are run only by the main agent after their prerequisites.
