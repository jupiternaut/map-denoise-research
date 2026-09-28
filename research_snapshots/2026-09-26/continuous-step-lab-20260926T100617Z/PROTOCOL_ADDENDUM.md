# Pre-evaluation implementation clarification

Written before the main evaluator has accessed the new experiment's reference.
The secondary quadratic vertex is not accepted from interpolation alone: its
actual image patches are re-scored, and it replaces the grid output only when
photo cost decreases on the identical source set. Extra scoring cost is recorded.
Steps on exactly KEEP routes or displacement <=1e-7mm are not photo-rescored;
these numerically stationary points retain their input coordinate. No new
threshold was chosen from geometry evaluation. The original protocol is retained
unchanged because the observation smoke already records its hash.

The reused R source IDs were selected in the prior experiment and already exposed
to selector development. They exclude original local A/B construction views but
are not fresh independent observations in this round. They now participate in
step construction; reporting them as independent confirmation would be wrong.
