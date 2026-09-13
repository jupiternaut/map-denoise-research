# replication: saved-output summary

396 outputs; 0 errors. Means over the stated synthetic cases, not real-scene accuracy.

| Group | Method | n | Normal MAE (mm) | 3D RMSE (mm) | Gap error (mm) | Mean ms |
|---|---|---:|---:|---:|---:|---:|
| replication | old:identity | 33 | 2.6700 | 3.0493 | 1.0087 | 0.01 |
| replication | old:xyz_mixture | 33 | 1.7342 | 2.5148 | 1.0989 | 10.89 |
| replication | old:joint_forced | 33 | 0.2139 | 0.3360 | 0.3191 | 12.89 |
| replication | old:open3d_icp_then_xyz | 33 | 1.1249 | 5.6565 | 2.3933 | 37.77 |
| replication | baseline:scalar_profile_hard | 33 | 0.1291 | 0.2807 | 0.0884 | 4.52 |
| replication | baseline:scalar_profile_lbfgs | 33 | 0.1377 | 0.2562 | 0.1393 | 11.89 |
| replication | proposal:framewise_proposal_hard | 33 | 0.1263 | 0.2764 | 0.1106 | 45.52 |
| replication | proposal:framewise_proposal_soft | 33 | 0.1379 | 0.2561 | 0.1406 | 45.43 |
| replication | proposal:framewise_proposal_global_pi | 33 | 0.2139 | 0.3360 | 0.3190 | 51.60 |
| replication | plane:plane_profile_map | 33 | 0.2051 | 0.3370 | 0.2258 | 41.38 |
| replication | normal:within_frame_normal_then_profile | 33 | 0.4801 | 0.6696 | 0.4288 | 4.79 |
| replication | normal:within_frame_normal_then_old | 33 | 0.4614 | 0.6116 | 0.6069 | 20.80 |
