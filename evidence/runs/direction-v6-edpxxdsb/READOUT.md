# V6 A: current-input direction replacement

PCA uses current observations only. Real numbers describe recovery to an original measured cloud, not independent geometry truth.
96/96 formal outputs; 24 existing real + 24 exposed synthetic inputs, two methods.

|Real intervention|Direction|XYZ recovery median mm|Normal mm|Tangent mm|Zero edit mm|Support mean|
|---|---|---:|---:|---:|---:|---:|
|zero|difference|0.717192|0.386438|0.346085|0.717192|0.678715|
|zero|pooled_pca|0.384549|0.384549|0.000000|0.384549|0.224652|
|normal_translation|difference|3.535175|3.499475|0.389485|0.717192|0.678090|
|normal_translation|pooled_pca|3.500000|3.500000|0.000860|0.384549|0.224444|
|tangent_translation|difference|3.463845|0.397943|3.421875|0.717192|0.678611|
|tangent_translation|pooled_pca|3.541968|0.385541|3.500000|0.384549|0.225069|
|small_rotation|difference|3.530700|3.492796|0.556922|0.717192|0.675660|
|small_rotation|pooled_pca|3.500000|3.493785|0.207428|0.384549|0.220035|

|Public synthetic subset|Direction|Surface MAE mean mm|Matched RMS mm|Same-XY gap error mm|
|---|---|---:|---:|---:|
|all|difference|0.205928|0.458664|0.514244|
|all|pooled_pca|0.213650|0.484050|0.505252|
|single|difference|0.061886|0.102692|NA|
|single|pooled_pca|0.061813|0.107161|NA|
|double|difference|0.253942|0.577322|0.514244|
|double|pooled_pca|0.264262|0.609680|0.505252|
|g2|difference|0.217729|0.350057|0.841128|
|g2|pooled_pca|0.217648|0.354799|0.841453|
|g4|difference|0.311201|0.818811|0.440585|
|g4|pooled_pca|0.314953|0.840915|0.397779|
|g8|difference|0.232897|0.563098|0.261018|
|g8|pooled_pca|0.260186|0.633325|0.276524|

All case-paired wins, ties and regressions are in AGGREGATES.json and RESULTS.csv.
Single-axis pointwise bounds are evaluator-only XYZ-to-measurement bounds; they are not normal-only or true-surface accuracy bounds.
The injection axis was defined using original-measurement PCA. Better recovery alone cannot validate the physical normal.
Direction replacement also recomputes bias, cells, assignments and support; this is a matched upstream intervention, not fixed-association attribution.
All outputs, actual support masks and separate movement masks are retained. The difference arm must exactly replay all 48 prior compatible outputs.
Method total 17.145s; wall 19.217s. Single CPU process under possible concurrent contention.
No new confirmation seeds, downloads, dependencies, GPU or algorithm edits in older versions.
