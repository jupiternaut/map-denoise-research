"""Frozen observation-only color membership and footprint-confidence proxy.

This module accepts grayscale images and an image-space reference anchor only.
It neither reads files nor accepts cameras, candidate depths, scene labels, or
ownership truth. Its binary outputs are uncalibrated confidence estimates.
"""

from collections import deque

import numpy as np


ESTIMATOR_VERSION = "fixed_color_component_erosion_v1"
INTENSITY_TOLERANCE = 20.0
PROTOTYPE_RADIUS = 1
FOOTPRINT_RADIUS = 1
REFERENCE_CONNECTIVITY = 4


def _reference_component(membership, anchor):
    """Select the four-connected component containing an eligible anchor."""
    x, y = anchor
    selected = np.zeros(membership.shape, dtype=bool)
    if not membership[y, x]:
        return selected
    selected[y, x] = True
    pending = deque([(y, x)])
    height, width = membership.shape
    while pending:
        yy, xx = pending.popleft()
        for ny, nx in ((yy - 1, xx), (yy + 1, xx), (yy, xx - 1), (yy, xx + 1)):
            if (0 <= ny < height and 0 <= nx < width
                    and membership[ny, nx] and not selected[ny, nx]):
                selected[ny, nx] = True
                pending.append((ny, nx))
    return selected


def _minimum_neighborhood(center):
    """Require all nine integer pixels; unknown image exterior is ineligible."""
    height, width = center.shape[-2:]
    padded = np.pad(center, ((0, 0), (1, 1), (1, 1)), constant_values=False)
    result = np.ones(center.shape, dtype=bool)
    for dy in range(2 * FOOTPRINT_RADIUS + 1):
        for dx in range(2 * FOOTPRINT_RADIUS + 1):
            result &= padded[:, dy:dy + height, dx:dx + width]
    return result


def estimate_fields(images, xy=(64, 64)):
    """Estimate center membership and conservative integer-pixel confidence.

    Parameters
    ----------
    images : numeric array-like, shape (3, H, W)
        Reference then two source grayscale images, in original brightness
        units. The fixed tolerance is 20 units; no per-image normalization is
        applied. Inputs must be finite. No supplied scene metadata is accepted.
    xy : pair of integers
        Reference-image (x, y) anchor, with its entire 3x3 patch in the image.

    Returns
    -------
    dict
        ``center`` and ``footprint`` are float64 arrays of shape (3, H, W)
        containing only 0 and 1; ``metadata`` describes the frozen procedure.
        Center membership compares every image to one prototype: the median of
        the reference 3x3 anchor patch. Only the reference mask is restricted to
        the anchor's four-connected component. Footprint confidence is the
        minimum of each integer pixel's 3x3 center-membership neighborhood.

    The footprint field is a conservative image-neighborhood proxy. It is not
    a recovered pixel integration fraction, an ownership probability calibrated
    to truth, or the full bilinear interpolation footprint at a noninteger
    coordinate. A downstream sampler must handle its contributing source pixels
    separately. Identical-looking surfaces and textured single planes can both
    defeat this estimator; an empty mask is preserved without fallback.
    """
    array = np.asarray(images)
    if array.ndim != 3 or array.shape[0] != 3 or min(array.shape[1:]) < 3:
        raise ValueError("images must have shape (3, H, W), with H and W >= 3")
    if array.dtype.kind not in "uif":
        raise TypeError("images must contain real numeric grayscale intensities")
    array = np.asarray(array, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("images must contain only finite intensities")
    anchor = np.asarray(xy)
    if (anchor.shape != (2,) or anchor.dtype.kind not in "uif"
            or not np.isfinite(anchor).all()
            or not np.equal(anchor, np.floor(anchor)).all()):
        raise ValueError("xy must be a finite integral (x, y) image coordinate")
    x, y = (int(value) for value in anchor)
    height, width = array.shape[-2:]
    if not (1 <= x < width - 1 and 1 <= y < height - 1):
        raise ValueError("xy must have a complete 3x3 reference patch in the image")

    prototype = float(np.median(array[0, y - 1:y + 2, x - 1:x + 2]))
    center = np.abs(array - prototype) <= INTENSITY_TOLERANCE
    center[0] = _reference_component(center[0], (x, y))
    footprint = _minimum_neighborhood(center)
    metadata = {
        "estimator_version": ESTIMATOR_VERSION,
        "prototype": prototype,
        "anchor_xy": [x, y],
        "prototype_radius": PROTOTYPE_RADIUS,
        "intensity_tolerance": INTENSITY_TOLERANCE,
        "reference_connectivity": REFERENCE_CONNECTIVITY,
        "source_component_restriction": "none",
        "footprint_radius": FOOTPRINT_RADIUS,
        "footprint_operator": "minimum_3x3_with_zero_exterior",
        "center_counts": [int(mask.sum()) for mask in center],
        "footprint_counts": [int(mask.sum()) for mask in footprint],
        "anchor_eligible": bool(center[0, y, x]),
        "calibrated_probability": False,
        "physical_integration_fraction": False,
    }
    return {
        "center": center.astype(np.float64),
        "footprint": footprint.astype(np.float64),
        "metadata": metadata,
    }
