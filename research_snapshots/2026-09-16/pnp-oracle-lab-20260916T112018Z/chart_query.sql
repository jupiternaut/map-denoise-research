SELECT family, regime, method, CAST(m AS INTEGER) AS m,
       100.0 * AVG(CAST(full_domain_error AS REAL)) AS error_percent,
       100.0 * AVG(CASE WHEN unseen_error <> '' THEN CAST(unseen_error AS REAL) END) AS unseen_error_percent,
       100.0 * AVG(CAST(train_error AS REAL)) AS train_error_percent,
       AVG(CAST(seen_unique AS REAL)) AS mean_unique_seen,
       COUNT(*) AS rows,
       COUNT(DISTINCT target_id) AS target_draws,
       COUNT(DISTINCT target_mask) AS unique_targets,
       COUNT(DISTINCT rep) AS sample_replicates_per_draw
FROM learning_rows
WHERE m = '32' AND regime = 'uniform'
GROUP BY family, regime, method, m
ORDER BY family, method;
