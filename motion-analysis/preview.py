"""Raw camera and raw LiDAR, rendered as movies beside the stick figure.

The results page shows three panels together: what the camera saw, what the
depth sensor saw, and what the tracker made of it. When a result looks wrong
that comparison is what tells you which of the three is at fault — a subject
who never appears in the depth movie is a capture problem, not a tracker one.

ORIENTATION IS THE WHOLE POINT HERE. The RGB video is always stored landscape
regardless of how the device was held, so a portrait recording needs a 90
degree clockwise rotation to stand the subject up, and the depth PNGs need the
identical turn. extract.py does this via sessiongeom.frame_geometry, and so
does this module — same call, same flag. Rendering a preview with its own idea
of rotation would show the subject upright in one panel and lying on a wall in
the next, which looks exactly like a tracking bug and is not one.

Both movies are written at the same output size so the three panels line up,
and both cover the same frames the tracker used (depth frame count, which is
what extract.py iterates over).
"""
import os
import sys

import cv2
import numpy as np


def _sibling(name):
    """Import a module from THIS directory by path, under a private key.

    Both motion-analysis and gait-analysis contain sessiongeom.py and
    skeleton3d.py. When the torso pipeline runs, gait-analysis's copies are
    already in sys.modules by the time this module loads, so a plain
    `import sessiongeom` would silently bind the wrong one — and
    gait-analysis/skeleton3d.py has no _reencode_h264 at all, so the import
    would fail outright. Loading by file path under a prefixed key lets both
    copies coexist in one interpreter.
    """
    import importlib.util
    key = f"_motion_{name}"
    if key in sys.modules:
        return sys.modules[key]
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), name + ".py")
    spec = importlib.util.spec_from_file_location(key, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod


sessiongeom = _sibling("sessiongeom")
_reencode_h264 = _sibling("skeleton3d")._reencode_h264

# Panel height in pixels. Depth is natively 192x256, so this upscales it; RGB is
# 1440x1920 and is downscaled to match. Both share an aspect ratio because the
# depth map covers the same field of view as the camera.
PANEL_H = 480

# Depth range is taken from the recording itself rather than fixed, because a
# hand close-up lives at 0.2-0.5 m and a gait recording at 1.5-4 m; one fixed
# scale would render one of them as a flat block of colour.
_SAMPLE_FRAMES = 40
_CLIP_PCT = (2.0, 98.0)


def _depth_files(folder):
    """Depth PNGs in frame order, filtered the way extract.py filters them.

    Hidden files matter: a Mac/iOS "Compress" leaves AppleDouble `._000000.png`
    entries that are not frames, and counting them shifts every frame index.
    """
    depth_dir = os.path.join(folder, "depth")
    return sorted(f for f in os.listdir(depth_dir)
                  if f.lower().endswith(".png") and not f.startswith("."))


def _open_writer(out_path, size, fps):
    """VideoWriter, trying H.264 first and falling back to mp4v.

    Same two-step as skeleton3d: this environment usually lacks the avc1
    encoder, and _reencode_h264 repairs the mp4v output afterwards.
    """
    for cc in ("avc1", "mp4v"):
        w = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*cc), fps, size)
        if w.isOpened():
            return w
        w.release()
    raise RuntimeError("could not open a video writer")


def _panel_size(frame_w, frame_h):
    """Output size at PANEL_H tall, width even (H.264 rejects odd dimensions)."""
    w = max(2, int(round(frame_w * PANEL_H / float(frame_h))))
    return (w + (w % 2), PANEL_H)


# Progress markers consumed by the web app's job runner. Emitted every
# _TICK frames so a long recording does not look stalled while the two
# previews render after the skeleton.
_TICK = 30


def _tick(label, done, total):
    if done % _TICK == 0:
        print(f"@@PREVIEW {label} {done}/{total}", flush=True)


def _finish(out_path, writer, label, n):
    writer.release()
    _reencode_h264(out_path)
    print(f"@@PREVIEW {label} {n}/{n}", flush=True)
    return out_path


def render_rgb(folder, out_path, fps=60.0, limit=None):
    """The camera's own video, rotated upright and trimmed to the LiDAR frames.

    Rendered from the native-resolution video rather than the 192x256 frame the
    tracker works in — same pixels, same orientation, just not thrown away.
    """
    geom = sessiongeom.frame_geometry(folder)
    rotate = geom["rotate"]
    n_lidar = len(_depth_files(folder))
    if limit is not None:
        n_lidar = min(n_lidar, limit)

    cap = cv2.VideoCapture(os.path.join(folder, "rgb.mp4"))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {os.path.join(folder, 'rgb.mp4')}")

    writer, size, n = None, None, 0
    try:
        while n < n_lidar:
            ret, frame = cap.read()
            if not ret:
                break
            if rotate:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            if writer is None:
                size = _panel_size(frame.shape[1], frame.shape[0])
                writer = _open_writer(out_path, size, fps)
            writer.write(cv2.resize(frame, size, interpolation=cv2.INTER_AREA))
            n += 1
            _tick("rgb", n, n_lidar)
    finally:
        cap.release()
    if writer is None:
        raise RuntimeError("no RGB frames could be read")
    return _finish(out_path, writer, "rgb", n)


def _depth_scale(folder, files):
    """(lo, hi) metres for the colour ramp, from a sample of the recording.

    Rotation is irrelevant here — percentiles over pixel values do not care how
    the image is turned — so the frames are sampled unrotated.
    """
    depth_dir = os.path.join(folder, "depth")
    step = max(1, len(files) // _SAMPLE_FRAMES)
    vals = []
    for fname in files[::step]:
        d = cv2.imread(os.path.join(depth_dir, fname), cv2.IMREAD_UNCHANGED)
        if d is None:
            continue
        v = d[d > 0]
        if v.size:
            vals.append(v.astype(np.float32) / 1000.0)
    if not vals:
        return 0.0, 1.0
    allv = np.concatenate(vals)
    lo, hi = (float(x) for x in np.percentile(allv, _CLIP_PCT))
    if hi - lo < 1e-3:
        hi = lo + 1e-3
    return lo, hi


def render_depth(folder, out_path, fps=60.0, limit=None):
    """The LiDAR depth maps as a colour movie, turned the same way as the RGB.

    Near is blue, far is red, and a pixel the sensor returned nothing for stays
    black — that black is informative, so it is deliberately not interpolated
    away. Nearest-neighbour upscaling for the same reason: the real resolution
    is 192x256 and smoothing it would imply detail that was never measured.
    """
    geom = sessiongeom.frame_geometry(folder)
    rotate = geom["rotate"]
    depth_dir = os.path.join(folder, "depth")
    files = _depth_files(folder)
    if limit is not None:
        files = files[:limit]
    if not files:
        raise RuntimeError(f"no depth PNGs in {depth_dir}")

    lo, hi = _depth_scale(folder, files)
    writer, size, n = None, None, 0
    try:
        for fname in files:
            d = cv2.imread(os.path.join(depth_dir, fname), cv2.IMREAD_UNCHANGED)
            if d is None:
                continue
            if rotate:
                d = cv2.rotate(d, cv2.ROTATE_90_CLOCKWISE)
            metres = d.astype(np.float32) / 1000.0

            norm = np.clip((metres - lo) / (hi - lo), 0.0, 1.0)
            img = cv2.applyColorMap((norm * 255).astype(np.uint8),
                                    cv2.COLORMAP_TURBO)
            img[metres <= 0] = 0          # no return from the sensor

            if writer is None:
                size = _panel_size(img.shape[1], img.shape[0])
                writer = _open_writer(out_path, size, fps)
            writer.write(cv2.resize(img, size, interpolation=cv2.INTER_NEAREST))
            n += 1
            _tick("lidar", n, len(files))
    finally:
        if writer is None:
            raise RuntimeError("no depth frames could be read")
    print(f"  depth colour range {lo:.2f}-{hi:.2f} m")
    return _finish(out_path, writer, "lidar", n)


def _frame_count(path):
    """Frames in a written movie, or None if it cannot be read back."""
    cap = cv2.VideoCapture(path)
    try:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    return n if n > 0 else None


def render_both(folder, rgb_path, depth_path, fps=60.0):
    """Render both previews, reporting failures rather than raising.

    Returns (rgb_path or None, depth_path or None). Callers are the web job and
    the CLI, where these two panels are a convenience: a session that cannot
    produce them — an unreadable rgb.mp4, a depth folder the sensor never
    filled — should still return its CSVs and its skeleton, which is the part
    the clinic actually needs.
    """
    try:
        rgb = render_rgb(folder, rgb_path, fps=fps)
    except Exception as exc:
        print(f"rgb preview skipped ({type(exc).__name__}: {exc})")
        rgb = None

    # Cap the depth movie to however many frames the camera actually yielded.
    # extract.py stops tracking when the RGB stream runs out, which on this
    # project's recordings is typically one frame before the depth folder ends —
    # so rendering every depth PNG would leave the LiDAR panel a frame longer
    # than the other two and drift out of sync at the end of playback.
    limit = _frame_count(rgb) if rgb else None
    try:
        depth = render_depth(folder, depth_path, fps=fps, limit=limit)
    except Exception as exc:
        print(f"lidar preview skipped ({type(exc).__name__}: {exc})")
        depth = None
    return rgb, depth


if __name__ == "__main__":
    import sys
    for arg in sys.argv[1:]:
        print(render_rgb(arg, os.path.join(arg, "rgb_preview.mp4")))
        print(render_depth(arg, os.path.join(arg, "lidar_preview.mp4")))
