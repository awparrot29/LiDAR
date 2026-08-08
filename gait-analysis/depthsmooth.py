"""Temporal smoothing of the per-joint LiDAR depth readings.

A single depth sample carries roughly 6 mm of sensor noise, measured on static
background pixels of a park_sim recording, and that noise dominates the
frame-to-frame jitter in the exported coordinates. Averaging each joint's depth
across neighbouring frames removes most of it.

The average is taken per joint, following the tracked point, rather than by
averaging the depth maps themselves. The subject is moving, so averaging maps
would mix body pixels with background pixels from adjacent frames and smear the
silhouette edges; following the joint averages the same anatomical point.

A centred window leaves constant-velocity motion untouched - the mean of a
straight line is the value at its midpoint - so only acceleration is slightly
flattened. At 60 FPS a 5-frame window spans 83 ms, short against the ~1 s of a
gait cycle.
"""
import numpy as np

# Frames averaged per reading, centred on the frame itself: 5 means the frame
# plus the two before and the two after. Must be odd so the window is centred.
DEFAULT_WINDOW = 5


def smooth_series(values, trusted, window=DEFAULT_WINDOW):
    """Centred moving average of `values`, using only the trusted samples.

    values  : (n,) depth readings in metres
    trusted : (n,) bool, False where the LiDAR confidence was zero or the
              reading was missing, so it should not pollute its neighbours
    window  : number of frames averaged, centred (forced odd)

    Untrusted positions still receive a value, taken from the trusted samples
    around them. Positions with no trusted sample anywhere in reach fall back to
    the nearest trusted reading, and if the joint was never seen at all the
    original values are returned unchanged.
    """
    values = np.asarray(values, dtype=float)
    trusted = np.asarray(trusted, dtype=bool)
    n = len(values)
    if n == 0:
        return values

    window = max(1, int(window))
    if window % 2 == 0:                  # keep it centred
        window += 1
    if window == 1 or not trusted.any():
        return values

    kernel = np.ones(window)
    weighted = np.where(trusted, values, 0.0)
    # 'same' zero-pads the ends, and dividing by the matching count of trusted
    # samples means the ends simply average over fewer frames rather than
    # being pulled toward zero.
    total = np.convolve(weighted, kernel, mode="same")
    count = np.convolve(trusted.astype(float), kernel, mode="same")

    out = np.divide(total, count, out=np.full(n, np.nan), where=count > 0)

    # Anywhere the window held nothing trusted, carry the nearest trusted value
    if np.isnan(out).any():
        idx = np.flatnonzero(trusted)
        nearest = idx[np.abs(np.subtract.outer(np.arange(n), idx)).argmin(axis=1)]
        out = np.where(np.isnan(out), values[nearest], out)
    return out


def smooth_joint_depths(depths, trusted, window=DEFAULT_WINDOW):
    """Apply smooth_series to {joint name: (n,) depths} with matching flags."""
    return {name: smooth_series(depths[name], trusted[name], window)
            for name in depths}
