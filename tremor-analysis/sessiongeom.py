"""Work out how a Stray Scanner session was held, and the matching geometry.

The RGB video is always recorded landscape (1920x1440) regardless of how the
device was held, so a session shot in portrait has the subject lying sideways in
the raw frames and needs a 90 degree rotation to stand them up. A session shot
in landscape is already upright and must NOT be rotated - doing so lays the
subject on their side, which shows up downstream as a figure standing on a wall.

Which case applies is recorded in imu.csv: gravity sits on the device y axis in
portrait and on the x axis in landscape.

This module also derives the camera intrinsics for the frame the pipeline
actually works in. The two are inseparable, because the 90 degree rotation swaps
the x and y axes of the camera matrix.
"""
import os

import cv2
import numpy as np

PORTRAIT = "portrait"
LANDSCAPE = "landscape"


def _mean_acceleration(folder):
    """Mean (a_x, a_y, a_z) from imu.csv, or None when unavailable.

    Units vary between Stray Scanner versions (m/s^2 in older recordings, g in
    newer ones); only the dominant axis matters here, so scale is irrelevant.
    """
    path = os.path.join(folder, "imu.csv")
    if not os.path.exists(path):
        return None
    total = np.zeros(3)
    count = 0
    # errors="ignore": some recordings carry stray bytes that break strict decoding
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        fh.readline()                       # header
        for line in fh:
            parts = line.split(",")
            if len(parts) < 4:
                continue
            try:
                total += (float(parts[1]), float(parts[2]), float(parts[3]))
            except ValueError:
                continue
            count += 1
    return total / count if count else None


def session_orientation(folder):
    """PORTRAIT or LANDSCAPE for the given session folder.

    Falls back to PORTRAIT when there is no usable imu.csv, which is how every
    session recorded before this check was handled.
    """
    mean = _mean_acceleration(folder)
    if mean is None:
        return PORTRAIT
    # Compare only the two in-plane axes; z is the screen normal
    return PORTRAIT if abs(mean[1]) >= abs(mean[0]) else LANDSCAPE


def frame_geometry(folder):
    """Geometry for the frame the tracker works in.

    Returns a dict with:
        orientation  PORTRAIT or LANDSCAPE
        rotate       True when frames need the 90 degree clockwise rotation
        width/height size the RGB frame is resized to, matching the depth maps
        fx, fy, cx, cy   intrinsics expressed in that frame
    """
    orientation = session_orientation(folder)

    # Depth maps define the working resolution: (rows, cols) = (height, width)
    depth_dir = os.path.join(folder, "depth")
    pngs = sorted(f for f in os.listdir(depth_dir)
                  if f.lower().endswith(".png") and not f.startswith("."))
    if not pngs:
        raise RuntimeError(f"No depth PNGs found in {depth_dir}")
    probe = cv2.imread(os.path.join(depth_dir, pngs[0]), cv2.IMREAD_UNCHANGED)
    if probe is None:
        raise RuntimeError(f"Could not read {pngs[0]} in {depth_dir}")
    depth_h, depth_w = probe.shape[:2]

    # Native RGB size, so the intrinsics can be scaled by the true factor rather
    # than hard-coded numbers
    cap = cv2.VideoCapture(os.path.join(folder, "rgb.mp4"))
    native_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    native_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1440
    cap.release()

    matrix = np.loadtxt(os.path.join(folder, "camera_matrix.csv"), delimiter=",")
    f_native = float(matrix[0, 0])
    cx_native, cy_native = float(matrix[0, 2]), float(matrix[1, 2])

    # Both axes shrink by the same factor: 1920->256 and 1440->192 are both 1/7.5
    scale = depth_w / float(native_w)
    fx = fy = f_native * scale
    cx, cy = cx_native * scale, cy_native * scale

    if orientation == PORTRAIT:
        # cv2.ROTATE_90_CLOCKWISE maps (u, v) -> (H - 1 - v, u), so the axes swap
        width, height = depth_h, depth_w
        fx, fy = fy, fx                     # equal for square pixels, kept explicit
        cx, cy = (depth_h - 1) - cy, cx
        rotate = True
    else:
        width, height = depth_w, depth_h
        rotate = False

    return {
        "orientation": orientation,
        "rotate": rotate,
        "width": width,
        "height": height,
        "fx": fx, "fy": fy, "cx": cx, "cy": cy,
        "native": (native_w, native_h),
        "depth": (depth_w, depth_h),
    }


if __name__ == "__main__":
    import sys
    for arg in sys.argv[1:]:
        g = frame_geometry(arg)
        print(f"{arg}\n  {g}")
