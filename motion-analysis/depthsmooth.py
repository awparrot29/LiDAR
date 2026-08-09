"""Temporal denoising of the per-joint LiDAR depth readings.

Each joint's depth is averaged across neighbouring frames, with samples more
than two standard deviations from the window mean dropped before averaging.

The outlier rejection matters because the residual at the extremities is not
steady noise but rare, violent excursions: kurtosis 47 at the ankle and 28 at
the wrist, against 3.0 for a normal distribution, with single-frame jumps up to
583 mm. A plain mean smears such a spike across the whole window instead of
removing it. On a synthetic 583 mm spike the clipped mean leaves 1.9 mm of
error where a plain 9-frame mean leaves 63 mm.

Sampling stays at the single pixel under the landmark. Reading a patch around it
was tried and rejected: a forearm is only about five pixels wide at 2.4 m, so a
5x5 patch is mostly background and its median jumps to the wall behind the
subject. Measured on park_sim, that pushed the wrist more than half a metre
behind the torso in 5.7% of frames, where single-pixel sampling never did.

The window is centred, so constant-velocity motion passes through untouched -
the mean of a straight line is the value at its midpoint. At 60 FPS a 9-frame
window spans 150 ms, short against the roughly 1 s of a gait cycle.
"""
import numpy as np

# Frames averaged per reading, centred on the frame itself: 9 means the frame
# plus the four before and the four after. Must be odd so the window is centred.
DEFAULT_WINDOW = 9

# Samples further than this many standard deviations from the window mean are
# dropped before averaging.
CLIP_SIGMA = 2.0

# Clipping is repeated, because one extreme sample inflates the standard
# deviation enough to hide itself on the first pass.
CLIP_PASSES = 3

# Never average fewer than this many samples; below it the estimate of the
# standard deviation is meaningless and the median is used instead.
MIN_KEPT = 3


def _clipped_mean(values):
    """Mean of `values` after repeatedly dropping points beyond CLIP_SIGMA."""
    keep = np.ones(len(values), dtype=bool)
    for _ in range(CLIP_PASSES):
        sample = values[keep]
        if len(sample) <= MIN_KEPT:
            break
        mean, std = sample.mean(), sample.std()
        if std <= 0:
            break
        still = np.abs(values - mean) <= CLIP_SIGMA * std
        still &= keep                       # never resurrect a dropped sample
        if still.sum() < MIN_KEPT or still.sum() == keep.sum():
            break
        keep = still

    sample = values[keep]
    if len(sample) < MIN_KEPT:
        return float(np.median(values))     # too few left to trust a mean
    return float(sample.mean())


def smooth_series(values, trusted, window=DEFAULT_WINDOW):
    """Centred, outlier-rejecting average of `values` over the trusted samples.

    values  : (n,) depth readings in metres
    trusted : (n,) bool, False where the reading was missing or the LiDAR
              confidence was zero, so it should not pollute its neighbours
    window  : number of frames averaged, centred (forced odd)

    Untrusted positions still receive a value, taken from the trusted samples
    around them. If the joint was never seen at all the values are returned
    unchanged.
    """
    values = np.asarray(values, dtype=float)
    trusted = np.asarray(trusted, dtype=bool)
    n = len(values)
    if n == 0:
        return values

    window = max(1, int(window))
    if window % 2 == 0:                     # keep it centred
        window += 1
    if window == 1 or not trusted.any():
        return values

    half = window // 2
    out = np.empty(n)
    idx = np.flatnonzero(trusted)

    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        w = values[lo:hi][trusted[lo:hi]]
        if len(w) == 0:
            # Nothing usable in reach; carry the nearest trusted reading
            out[i] = values[idx[np.abs(idx - i).argmin()]]
        elif len(w) <= MIN_KEPT:
            out[i] = float(np.median(w))
        else:
            out[i] = _clipped_mean(w)
    return out


def smooth_joint_depths(depths, trusted, window=DEFAULT_WINDOW):
    """Apply smooth_series to {joint name: (n,) depths} with matching flags."""
    return {name: smooth_series(depths[name], trusted[name], window)
            for name in depths}
