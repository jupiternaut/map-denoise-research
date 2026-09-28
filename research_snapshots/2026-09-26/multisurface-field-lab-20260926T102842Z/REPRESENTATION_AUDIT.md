# Reference-free representation audit

Status: PASS. All 60 cases were read only after all three scene seals existed.
No laser, evaluation support, native parent, geometry errors, photograph pixels or sparse 3D points were read.

| Condition | K2 ray gap median / p90 (mm) | Gap < 0.1 mm | Field outside old interval K1 / K2a / K2b | Raw >6mm clip rate K1 / K2a / K2b |
|---|---:|---:|---:|---:|
| native | 4.5925 / 12.0000 | 25.01% | 89.26%/84.64%/85.81% | 68.54%/57.19%/56.16% |
| minus1 | 4.2402 / 12.0000 | 23.73% | 84.43%/79.45%/78.59% | 68.50%/56.79%/56.70% |
| plus1 | 4.7866 / 12.0000 | 25.17% | 85.58%/80.48%/80.33% | 68.57%/56.98%/56.47% |
| minus3 | 3.2284 / 12.0000 | 25.54% | 84.41%/79.65%/77.62% | 68.66%/57.43%/57.01% |
| plus3 | 4.6051 / 12.0000 | 22.81% | 85.75%/80.65%/80.53% | 68.80%/57.29%/56.40% |

Counts pool eligible rows; they are not the scene-balanced geometry-quality score.
Gap is the clipped separation along a reference-camera ray, not physical layer thickness.
Layer IDs are local to patches; JSON also reports near/far-ordered posterior mass.
Every supported core prediction was independently reconstructed from its patch coefficients,
converted through original camera metadata, clipped to ±6mm, and checked against saved POOL geometry.
Maximum reconstructed world-coordinate discrepancy: 0 mm.
Unknown evidence and unsupported fits retain input exactly; unsupported output labels are KEEP.
These checks establish a real shared-parameter implementation, not correct physical surface identity or improved accuracy.
