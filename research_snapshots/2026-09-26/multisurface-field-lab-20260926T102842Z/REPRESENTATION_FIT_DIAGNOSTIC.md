# Fixed-case field-fit diagnostic

Case: `scan55_roi0__native` (fixed before looking at geometry quality).
No GT, evaluation results or native-parent geometry were read. All fields below are reconstructed before ±6 mm clipping.

| Quantity | Median (mm) | p90 (mm) | >3 mm | >6 mm |
|---|---:|---:|---:|---:|
| K1_input_to_field_ray_mm | 33.5550 | 73.8343 | 95.50% | 90.74% |
| K2_individual_layer_input_distance_ray_mm | 29.2242 | 122.5608 | 76.89% | 71.07% |
| K2_input_to_nearest_field_ray_mm | 3.9147 | 28.8142 | 53.92% | 42.53% |
| K2_best_photo_candidate_to_nearest_field_ray_mm | 5.4874 | 29.3555 | 61.80% | 47.19% |
| K2_any_candidate_to_any_field_ray_mm | 0.8745 | 22.8142 | 34.64% | 29.72% |
| K2_input_to_nearest_field_rho_mm | 3.6752 | 28.2688 | 52.89% | 42.08% |

K2 clipping by row: neither layer 0.39%; exactly one 57.09%; both 42.53%.

The percentage of clipped individual layers is not the percentage of points with no nearby layer.
Any-candidate proximity is more permissive than matching the best photometric candidate, and neither establishes true geometry.
Patch depth/rho spans and all per-patch residuals are preserved in the JSON; patch-span summaries weight patches equally.
Median core depth span: 127.4951 mm; median halo depth span: 173.4453 mm.
Independent coefficient reconstruction vs saved clipped offsets: maximum difference 0 mm.
This diagnostic can expose coarse patch/model mismatch or an unfavorable local fit, but does not distinguish those causes or certify a global optimum.
