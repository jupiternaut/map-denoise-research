# DA support rejection diagnostic

Saved PCA basis and scan bias, unchanged current XYZ, unchanged local fit/gates. No evaluator truth loaded.

|Case|Cells|Median-residual range mm|Slope-norm range|Residual-failed cells|Slope-failed cells|
|---|---:|---:|---:|---:|---:|
|da_junction__zero|4|31.677–36.462|0.4582–0.7614|4|4|
|da_junction__normal_translation|4|31.721–36.573|0.1737–0.9032|4|3|
|da_thin__zero|5|18.958–26.923|49.0685–85.6210|5|5|
|da_thin__normal_translation|5|18.436–26.599|48.7698–84.6631|5|5|
|da_wall__zero|8|22.495–38.395|0.0330–1.5084|8|5|
|da_wall__normal_translation|8|22.491–38.422|0.0339–1.4802|8|6|

All six actual support masks and identity outputs are reproduced. Thresholds remain median absolute residual <= 5 mm, slope norm <= 0.25, point absolute residual <= 8 mm and confidence >= 0.8.
Raw point-pass counts in CELLS.csv are diagnostic, not newly accepted points. Failed cell-level criteria veto them.
This identifies mismatch with the existing local model/gates at supplied sigma=2 mm. It does not distinguish true geometry, noise miscalibration, association error, or bad global direction by itself.
No gate was relaxed; no parameter, existing run, estimator source, or point output was changed.
