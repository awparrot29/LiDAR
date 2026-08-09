"""Two-view 3D stick figure, so the coordinates can be verified by eye.

The front view is what the camera saw. The side view is rotated 90 degrees about
the vertical axis, and that is the one worth watching: if the reconstruction were
flat or wrong, the side view would collapse toward a line. Real depth structure
shows up there — a fist reads differently from an open hand, and a walking figure
moves through the frame rather than sliding up it.

Drawn with cv2 rather than matplotlib because it runs once per frame; this is
roughly two orders of magnitude faster and needs no display.

Bounds are computed once over the whole recording and held fixed, so the figure
moves inside a stationary frame instead of the axes chasing it — the fix for the
"stick figure appears to walk upward" problem.
"""
import os

import cv2
import numpy as np

PANEL = 470
MARGIN = 0.10
HEADER = 28
BONE_COLOR = (200, 130, 50)      # BGR
JOINT_COLOR = (70, 220, 240)
ROOT_COLOR = (60, 60, 240)
BG = 26


def _to_array(arrays, names):
    """(T, n, 3) with untracked frames as NaN."""
    P = np.stack([np.asarray(arrays[n], float) for n in names], axis=1)
    P[np.all(P == 0, axis=2)] = np.nan
    return P


def _bounds(P, pct=1.0):
    """Fixed, isotropic bounds over the finite points.

    Percentiles rather than min/max: a handful of frames with bad depth put
    points metres away, and using the extremes would zoom out until the real
    figure is a dot. Measured on the hand recording, min/max bounds shrank the
    hand to roughly a tenth of the panel.
    """
    flat = P.reshape(-1, 3)
    flat = flat[np.all(np.isfinite(flat), axis=1)]
    if not len(flat):
        return None
    lo = np.percentile(flat, pct, axis=0)
    hi = np.percentile(flat, 100.0 - pct, axis=0)
    span = float(np.max(hi - lo)) or 0.1
    return (hi + lo) / 2.0, span * (0.5 + MARGIN)


def _pt(p, mid, pad, h, v):
    u = (p[h] - mid[h]) / (2 * pad) + 0.5
    w = (p[v] - mid[v]) / (2 * pad) + 0.5
    return int(u * PANEL), int(w * PANEL)


def _panel(pts, bones_idx, mid, pad, h, v, title):
    img = np.full((PANEL, PANEL, 3), BG, np.uint8)
    cv2.rectangle(img, (0, 0), (PANEL - 1, PANEL - 1), (70, 70, 70), 1)
    cv2.putText(img, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                (225, 225, 225), 1, cv2.LINE_AA)
    for a, b in bones_idx:
        if np.all(np.isfinite(pts[a])) and np.all(np.isfinite(pts[b])):
            cv2.line(img, _pt(pts[a], mid, pad, h, v),
                     _pt(pts[b], mid, pad, h, v), BONE_COLOR, 2, cv2.LINE_AA)
    for i, p in enumerate(pts):
        if np.all(np.isfinite(p)):
            cv2.circle(img, _pt(p, mid, pad, h, v), 4,
                       ROOT_COLOR if i == 0 else JOINT_COLOR, -1, cv2.LINE_AA)
    return img


def _reencode_h264(path):
    """Re-encode to H.264 in place, if an ffmpeg binary can be found.

    OpenCV in this environment cannot open the avc1 encoder (libopenh264 is
    missing), so it falls back to mp4v. That plays in VLC but not in browsers or
    Windows Photos, and is several times larger. imageio_ffmpeg ships a static
    ffmpeg, so use it when present. Silently leaves the mp4v file alone on any
    failure — a playable-somewhere video beats no video.
    """
    try:
        import subprocess
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return False
    tmp = path + ".h264.mp4"
    try:
        r = subprocess.run(
            [exe, "-y", "-loglevel", "error", "-i", path,
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
             "-movflags", "+faststart", tmp],
            capture_output=True, timeout=600)
        if r.returncode == 0 and os.path.exists(tmp) and os.path.getsize(tmp) > 0:
            os.replace(tmp, path)
            return True
    except Exception:
        pass
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
    return False


def render(arrays, profile, out_path, fps=60.0):
    """Write a side-by-side front/side stick figure video. Returns the path."""
    names = list(profile["landmarks"])
    index = {n: i for i, n in enumerate(names)}
    bones_idx = [(index[a], index[b]) for a, b in profile["bones"]
                 if a in index and b in index]

    P = _to_array(arrays, names)
    b = _bounds(P)
    if b is None:
        raise RuntimeError("no finite 3D points to render")
    mid, pad = b

    W, H = PANEL * 2, PANEL + HEADER
    writer = None
    for cc in ("avc1", "mp4v"):
        w = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*cc), fps, (W, H))
        if w.isOpened():
            writer = w
            break
        w.release()
    if writer is None:
        raise RuntimeError("could not open a video writer")

    try:
        for t in range(P.shape[0]):
            frame = np.full((H, W, 3), BG, np.uint8)
            # y is down in camera coordinates, so it maps straight to image rows
            frame[HEADER:, :PANEL] = _panel(P[t], bones_idx, mid, pad, 0, 1,
                                            "FRONT (as camera saw it)")
            frame[HEADER:, PANEL:] = _panel(P[t], bones_idx, mid, pad, 2, 1,
                                            "SIDE (rotated 90 deg)")
            cv2.putText(frame, f"{profile['kind']}   frame {t+1}/{P.shape[0]}   "
                               f"{t / fps:5.2f}s", (10, 19),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (175, 175, 175), 1,
                        cv2.LINE_AA)
            writer.write(frame)
    finally:
        writer.release()
    if _reencode_h264(out_path):
        print("  (re-encoded to H.264 so it plays in any player or browser)")
    return out_path
