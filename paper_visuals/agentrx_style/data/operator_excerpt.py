    anchor = quadratic[control_k]
    normals = anchor["normal"]
    delta_vectors = np.stack([quadratic[k]["correction"][:, None] * quadratic[k]["normal"]
                              for k in actual_ks], axis=1)
    projected = np.einsum("nsi,ni->ns", delta_vectors, normals)
    point_variance = np.median(np.stack([quadratic[k]["noise_variance"] for k in actual_ks], axis=1), axis=1)
    prediction_variances = np.stack([quadratic[k]["prediction_variance"] for k in actual_ks], axis=1)
    # Numerical floor is dimensional and tiny; it is not a sensor-noise guess.
    floor = np.maximum(anchor["tangent_scale"]**2 * 1e-24, np.finfo(float).tiny)
    safe_variances = np.maximum(prediction_variances, floor[:, None])
    # Equivalent inverse-variance ratios, without overflow for identical points.
    precision = safe_variances.min(axis=1, keepdims=True) / safe_variances
    weights = precision / precision.sum(axis=1, keepdims=True)
    mean_delta = np.sum(weights * projected, axis=1)
    mean_vector = mean_delta[:, None] * normals
    disagreement = np.sum(weights * np.sum((delta_vectors - mean_vector[:, None])**2, axis=2), axis=1)
    # Fits share observations; do not claim a 1/number_of_scales variance gain.
    prediction_variance = np.sum(weights * prediction_variances, axis=1)
    alpha = point_variance / (point_variance + prediction_variance + disagreement + floor)
    consensus = points + (alpha * mean_delta)[:, None] * normals
