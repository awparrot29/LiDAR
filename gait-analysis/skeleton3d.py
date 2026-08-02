"""Render a 3D skeleton movie from the joint coordinate CSVs.

Reads the per-landmark CSVs written by calculateangle.py (charts/<session>/data)
and animates the landmarks in 3D over time, drawing the bones between them.
Writes an MP4 via OpenCV, so no ffmpeg binary is required.
"""
import os

import matplotlib
matplotlib.use("Agg")           # headless — must be set before pyplot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)
import cv2
import numpy as np

# The twelve landmarks calculateangle.py writes a coordinate CSV for
LANDMARKS = [
    "left shoulder", "right shoulder",
    "left elbow", "right elbow",
    "left wrist", "right wrist",
    "left hip", "right hip",
    "left knee", "right knee",
    "left ankle", "right ankle",
]

# Pairs of landmarks joined by a bone
BONES = [
    ("left shoulder", "right shoulder"),    # shoulder girdle
    ("left hip", "right hip"),              # pelvis
    ("left shoulder", "left hip"),          # torso
    ("right shoulder", "right hip"),
    ("left shoulder", "left elbow"),        # arms
    ("left elbow", "left wrist"),
    ("right shoulder", "right elbow"),
    ("right elbow", "right wrist"),
    ("left hip", "left knee"),              # legs
    ("left knee", "left ankle"),
    ("right hip", "right knee"),
    ("right knee", "right ankle"),
]

LEFT_COLOR = "#e8543f"
RIGHT_COLOR = "#3f8ae8"
SPINE_COLOR = "#9aa4b2"


def _bone_color(a, b):
    if a.startswith("left") and b.startswith("left"):
        return LEFT_COLOR
    if a.startswith("right") and b.startswith("right"):
        return RIGHT_COLOR
    return SPINE_COLOR


def load_landmarks(data_dir):
    """Load {landmark name: (n_frames, 3) array} from a charts data folder."""
    out = {}
    for name in LANDMARKS:
        path = os.path.join(data_dir, f"{name}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing coordinate CSV: {path}")
        arr = np.loadtxt(path, delimiter=",")
        if arr.ndim == 1:                    # single-frame session
            arr = arr.reshape(1, 3)
        out[name] = arr
    n = min(len(a) for a in out.values())
    return {k: v[:n] for k, v in out.items()}


def _to_plot_space(arr):
    """Camera coords (x right, y down, z depth) -> plot coords (x, depth, up)."""
    return np.column_stack([arr[:, 0], arr[:, 2], -arr[:, 1]])


def render_movie(landmarks, out_path, fps=60, size=(720, 540), elev=12, azim=-70,
                 progress_every=10):
    """Animate the landmarks in 3D and write an MP4 to out_path."""
    pts = {k: _to_plot_space(v) for k, v in landmarks.items()}
    n_frames = len(next(iter(pts.values())))
    stacked = np.concatenate(list(pts.values()), axis=0)

    # Drop the (0,0,0) sentinels the tracker emits when a joint is never seen,
    # so they don't stretch the axes to include the origin.
    real = stacked[~np.all(np.isclose(stacked, 0.0), axis=1)]
    finite = real[np.isfinite(real).all(axis=1)] if len(real) else stacked

    # One cubic bounding box keeps the body from being distorted by autoscaling
    lo, hi = finite.min(axis=0), finite.max(axis=0)
    centre = (lo + hi) / 2.0
    span = float(np.max(hi - lo))
    if not np.isfinite(span) or span <= 0:
        span = 1.0
    half = span / 2.0 * 1.15

    w, h = size
    dpi = 100
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlim(centre[0] - half, centre[0] + half)
    ax.set_ylim(centre[1] - half, centre[1] + half)
    ax.set_zlim(centre[2] - half, centre[2] + half)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Depth from camera (m)")
    # Negative values are normal here: the camera axis origin sits above the
    # subject, so this is height relative to the camera, not to the floor.
    ax.set_zlabel("Height rel. camera (m)")
    ax.set_title("3D Joint Skeleton")

    # Build the artists once and only update their data each frame — redrawing
    # from scratch per frame is several times slower.
    bone_lines = []
    for a, b in BONES:
        line, = ax.plot([], [], [], "-", lw=2.2, color=_bone_color(a, b))
        bone_lines.append((a, b, line))
    joint_dots, = ax.plot([], [], [], "o", ms=4.5, color="#f0f6fc",
                          markeredgecolor="#30363d", linestyle="None")
    label = ax.text2D(0.02, 0.95, "", transform=ax.transAxes, fontsize=9,
                      color="#57606a")

    writer = None
    try:
        for i in range(n_frames):
            for a, b, line in bone_lines:
                pa, pb = pts[a][i], pts[b][i]
                line.set_data([pa[0], pb[0]], [pa[1], pb[1]])
                line.set_3d_properties([pa[2], pb[2]])

            frame_pts = np.array([pts[name][i] for name in LANDMARKS])
            joint_dots.set_data(frame_pts[:, 0], frame_pts[:, 1])
            joint_dots.set_3d_properties(frame_pts[:, 2])
            label.set_text(f"frame {i + 1}/{n_frames}   t = {i / 60.0:5.2f}s")

            fig.canvas.draw()
            rgba = np.asarray(fig.canvas.buffer_rgba())
            bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)

            if writer is None:
                fh, fw = bgr.shape[:2]
                writer = cv2.VideoWriter(out_path,
                                         cv2.VideoWriter_fourcc(*"mp4v"),
                                         fps, (fw, fh))
                if not writer.isOpened():
                    raise RuntimeError("Could not open the MP4 writer")
            writer.write(bgr)

            if i % progress_every == 0:
                # Progress marker consumed by the web app's job runner
                print(f"@@RENDER {i}/{n_frames}", flush=True)
    finally:
        if writer is not None:
            writer.release()
        plt.close(fig)

    return out_path


def main(folder=None, out=None):
    if folder is None:
        folder = input("Folder name: ")
    if out is None:
        out = input("Output file name (.mp4): ")
    data_dir = os.path.join("charts", folder, "data")
    render_movie(load_landmarks(data_dir), out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
