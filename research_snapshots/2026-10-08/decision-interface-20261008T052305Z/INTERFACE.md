# Implementation interface (frozen before observations)

The sealed predictor remains unchanged at ../mixed-pixel-20261008T041249Z/predictor.py. Import it using importlib and keep old directories read-only.

decisions.py owns pure functions. Public API:
- decide(grid, loss, candidates, incumbent, *, rule, sigma2, temperature=1., raw_valid=True, accepted=None, historical_intervals=None, kappa=1.) -> JSON-safe dict with selected_depth, reason, move, scale_valid, accepted_count, raw_valid, flat; rules P/M/Mraw/R/U. If accepted supplied (full9) it overrides 9/4 acceptance. P may receive historical_intervals for exact replay. Legacy P ties remain original; other rules deduplicate/sort actions and KEEP on ties.
- acceptance(grid, loss, sigma2) -> Boolean vector.
- distribution(grid, loss, sigma2, temperature, accepted, uniform=False) -> q array, using physical depth cell widths.
- crps(grid, q, truth) -> scalar.
- combine_scale(sigma_values, pixel_counts, sigma_cal2=None) -> (sigma2, per_fold_sources). Valid local values >=1 remain unchanged; invalid only falls back.

new_fixtures.py owns independent 7x7 renderer and create_dataset(root, stage), stage='calibration' or 'confirmation'. Produces root/observed/inputs.json rows {id,cameras,image_file,sha256}; root/truth/metadata.json rows {id,true_depth,mechanism,producer_parameters}. No hidden labels/seeds in observed. Calibration additionally scores the known true depth via separate runner but never supplies it to auxiliary fitting. Confirmation observations can contain incumbent/candidates only in a subsequently prepared task manifest.

Root runner owns scoring, calibration, locks, B2 replay, evaluation and B3 conditional execution. B3 is not generated unless the B2 gate passes. Action grid450:15:900 plus incumbent; score grid450:2:900 plus actions (calibration also plus known calibration depth for computing ell_s). Evaluation source truth only after prediction seal.
