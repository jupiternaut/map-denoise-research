# Shared implementation interface

Images: numpy float64 (3,128,128), gray units0..255. Camera: dict K(3,3),R(3,3),C(3,) with world-to-camera rotation. Reference optical depth scan np.arange(300,1001,2). Query (64,64). Cameras and images exported separately from world truth. No global truth constants in predictor.py.

predictor.py (model agent):
- estimate_aux(ref, train, ref_cam, train_cam, params) -> dict (numpy arrays allowed). No source scoring view/candidates/truth arguments.
- predict_curve(aux, ref_cam, eval_cam, eval_image, grid, incumbent, params, mode='dynamic'|'fixed') -> dict with loss (len(grid)), sigma, valid, reason, pixel_count, optionally diagnostics. Fixed raw sensor ROI identical across arms/depth, derived from public ref_window + cameras + grid and used by both. No partial score comparisons for invalid hypotheses.
- Aux shared minimum fields: mode='two'|'single'|'unavailable'; foreground/background as (128,128) reference-coordinate appearance fields; mask (128,128) reference occupancy fallback; optional polygon vertices in reference pixel coordinates for exact boundary; background_depth; sigma; valid; metadata. For mode='single', mask=all ones. Sampling arrays uses order1 and frozen nearest extrapolation if needed, never refits on eval pixels.
- Oracle aux uses exactly same fields and predictor, but separate producer may set more accurate appearances/polygon/background. Need auxiliary_valid and sigma_valid separated if possible: raw oracle loss can run even if ordinary sigma unavailable. Runner applies shared E-derived sigma to all four arms' P endpoint, marking unavailable sigma distinctly.

fixtures.py (generator agent):
- create_e1(output_dir) -> write observed/inputs.json with id,cameras,image_file/hash; observed/*.npz(images only); truth/metadata.json with group, mechanism, background_pair, seed,true_depth; oracle/*.npz + oracle/manifest.json containing aux for predictor. Do not import predictor to render input images.
- 24 base worlds=4 mechanisms(flat_contrast,flat_equal,textured_boundary,textured_single) x widths(1.65,2.65 pixels) x seeds1103,2207,3301. Add background alternative for flat_contrast/textured_boundary only ->36 worlds. Main true depth600, background900; fixed candidates handled by runner. Actual cameras as prior f160,C±60,Ry±.07@Rz±.11. Nine-ray pixel integration from continuous geometry and appearance. Random phases affect textures but not method input IDs.
- Oracle appearances: true radiance pulled into reference-coordinate fields; polygon corresponds true reference silhouette (NOT truth target depth supplied to predictor). Background_depth allowed oracle privileged. npz fields mode (unicode),foreground,background,mask,polygon (empty when absent),background_depth,valid; sigma from E runner not truth.
- e0.py standalone command writes E0 numerical/analytic edge checks under e0/, independent of E1 and no import predictor required until checking its low-level API. Analytic half-plane/pixel area vs high-resolution samples, positive/negative mixed-color tests and fixed coverage ablation with distinct texture-warp test. E0 is sanity, not a new algorithm score.

baseline.py (baseline agent):
- full9_intervals(images,cameras,grid) -> dict intervals + raw scores/accepted, uses exact historical scoring rule through read-only definitions.
- choose(intervals,candidates,incumbent) -> dict selected_depth,support_mean,estimated_squared_gain, matches oldP.
- replay_legacy(output_dir) -> reproduce full9 30 archived image inputs with incumbent900 and five historical candidates; independently compare curves/decisions/summary to old records. Reads historical files but never imports renderer into ordinary stages.

All agents: coordinate required interface changes with main before adopting. Do not run full E1 evaluation before main's lock. Unit tests synthetic examples are allowed. No figures library installation.
