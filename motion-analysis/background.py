"""Static background model, used to spot depth readings that are the scene.

A joint reading is sometimes not the joint. The LiDAR map is 192x256, so a thin
limb against a distant wall can fail to register and the pixel under the
landmark returns whatever is behind the subject instead. Averaging such a
reading with its neighbours does not help - it is not noise, it is the wrong
surface.

The scene behind the subject is static, so it can be learned from the recording
itself. For each pixel, the background is the farthest surface that pixel ever
credibly shows. A reading that sits at that distance is the background, not the
subject, and is marked untrusted so the temporal filter fills it in from the
frames either side.

The background is estimated with a high percentile rather than the median. The
median is contaminated wherever the subject lingers over a pixel - measured on
park_sim, a median model put four of the five test joints BEHIND their own
"background", which would make every reading there look valid.

This assumes the camera does not move. `camera_drift` reports how far it did, so
a handheld recording can be detected.
"""
import os

import cv2
import numpy as np

# Percentile of each pixel's depth history taken as the background. High enough
# to ignore the subject even when they occupy a pixel for most of the clip,
# below 100 so a single spurious far reading does not define the surface.
BACKGROUND_PCT = 90

# A reading closer than this to the modelled background is treated as the
# background. Must stay under the smallest genuine joint-to-background gap; the
# ankle is the tightest case, sitting about 195 mm in front of the floor with a
# 5th percentile of 69 mm.
BACKGROUND_TOLERANCE_M = 0.05

# Frames sampled to build the model. A few hundred is ample for a percentile,
# and keeps the working set small enough for a 1 GB server.
MAX_MODEL_FRAMES = 200

# Drift beyond this suggests the camera was not on a stand and the model cannot
# be trusted.
DRIFT_WARN_M = 0.15


def camera_drift(folder):
    """Largest distance the camera moved from its starting pose, in metres.

    Returns None when the session has no usable odometry.
    """
    path = os.path.join(folder, "odometry.csv")
    if not os.path.exists(path):
        return None
    try:
        data = np.genfromtxt(path, delimiter=",", skip_header=1)
    except (OSError, ValueError):
        return None
    if data.ndim != 2 or data.shape[1] < 5 or len(data) < 2:
        return None
    xyz = data[:, 2:5]
    xyz = xyz[np.isfinite(xyz).all(axis=1)]
    if len(xyz) < 2:
        return None
    return float(np.max(np.linalg.norm(xyz - xyz[0], axis=1)))


def build_model(depth_folder, depth_files, rotate, max_frames=MAX_MODEL_FRAMES):
    """Per-pixel background depth in metres, or None if it cannot be built."""
    if len(depth_files) < 10:
        return None

    step = max(1, len(depth_files) // max_frames)
    chosen = depth_files[::step]

    frames = []
    for name in chosen:
        d = cv2.imread(os.path.join(depth_folder, name), cv2.IMREAD_UNCHANGED)
        if d is None:
            continue
        if rotate:
            d = cv2.rotate(d, cv2.ROTATE_90_CLOCKWISE)
        frames.append(d)
    if len(frames) < 10:
        return None

    stack = np.stack(frames).astype(np.float32) / 1000.0
    # Zeros are dropouts, not surfaces, so they must not drag the percentile down
    stack[stack <= 0] = np.nan
    with np.errstate(invalid="ignore"):
        model = np.nanpercentile(stack, BACKGROUND_PCT, axis=0)
    # Pixels that never returned anything cannot reject anything
    model[~np.isfinite(model)] = np.inf
    return model


def is_background(model, depth, u, v, tolerance=BACKGROUND_TOLERANCE_M):
    """True when a reading at (u, v) is close enough to the background to be it."""
    if model is None or depth <= 0:
        return False
    behind = model[v, u]
    if not np.isfinite(behind):
        return False
    return (behind - depth) < tolerance
