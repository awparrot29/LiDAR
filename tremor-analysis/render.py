"""Two-view 3D skeleton video, so the reconstruction can be eyeballed.

Front view is what the camera sees. The side view is rotated 90 degrees about the
vertical axis, which is the check that matters: if the reconstruction were flat or
fake, the side view would collapse to a line. Real depth structure shows up there.

Drawn with cv2 rather than matplotlib because it has to run once per frame and
this is roughly two orders of magnitude faster.
"""
import cv2
import numpy as np

PANEL = 480
MARGIN = 0.12
BONE = (200, 120, 40)          # BGR
JOINT = (60, 220, 240)
ROOT = (60, 60, 240)


def _bounds(cam):
    """Symmetric world bounds over all finite frames, so the view never jitters."""
    P = cam.reshape(-1, 3)
    P = P[np.all(np.isfinite(P), axis=1)]
    if not len(P):
        return None
    lo, hi = P.min(axis=0), P.max(axis=0)
    span = float(np.max(hi - lo))
    if span <= 0:
        span = 0.1
    mid = (hi + lo) / 2.0
    pad = span * (0.5 + MARGIN)
    return mid, pad


def _project(pt, mid, pad, ax_h, ax_v, flip_v=True):
    """World point -> panel pixel, orthographic, fixed scale."""
    u = (pt[ax_h] - mid[ax_h]) / (2 * pad) + 0.5
    v = (pt[ax_v] - mid[ax_v]) / (2 * pad) + 0.5
    if not flip_v:
        v = 1.0 - v
    return int(u * PANEL), int(v * PANEL)


def _draw(panel, pts, conns, mid, pad, ax_h, ax_v, title):
    cv2.rectangle(panel, (0, 0), (PANEL - 1, PANEL - 1), (70, 70, 70), 1)
    cv2.putText(panel, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (230, 230, 230), 1, cv2.LINE_AA)
    if conns:
        for a, b in conns:
            if np.all(np.isfinite(pts[a])) and np.all(np.isfinite(pts[b])):
                cv2.line(panel, _project(pts[a], mid, pad, ax_h, ax_v),
                         _project(pts[b], mid, pad, ax_h, ax_v), BONE, 2,
                         cv2.LINE_AA)
    for i, p in enumerate(pts):
        if np.all(np.isfinite(p)):
            cv2.circle(panel, _project(p, mid, pad, ax_h, ax_v), 4,
                       ROOT if i == 0 else JOINT, -1, cv2.LINE_AA)


def two_view_video(track, out_path, fps=None):
    cam = track["cam"]
    conns = track["connections"]
    b = _bounds(cam)
    if b is None:
        raise RuntimeError("no finite 3D points to render")
    mid, pad = b
    fps = fps or track["geom"]["fps"]

    W, H = PANEL * 2, PANEL + 30
    writer = None
    for fourcc in ("avc1", "mp4v"):
        w = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*fourcc), fps, (W, H))
        if w.isOpened():
            writer = w
            break
        w.release()
    if writer is None:
        raise RuntimeError("could not open a video writer")

    try:
        for t in range(cam.shape[0]):
            frame = np.full((H, W, 3), 24, np.uint8)
            left = frame[30:, :PANEL]
            right = frame[30:, PANEL:]
            # x horizontal / y vertical = as the camera sees it
            _draw(left, cam[t], conns, mid, pad, 0, 1, "FRONT (camera view)")
            # z horizontal / y vertical = rotated 90 deg about vertical
            _draw(right, cam[t], conns, mid, pad, 2, 1, "SIDE (rotated 90 deg)")
            cv2.putText(frame, f"frame {t}/{cam.shape[0]}   "
                              f"{t / fps:.2f}s", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1,
                        cv2.LINE_AA)
            writer.write(frame)
    finally:
        writer.release()
    return out_path
