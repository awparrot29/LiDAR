"""Render a 3D stick figure movie from the joint coordinate CSVs.

Reads the per-landmark CSVs written by calculateangle.py (charts/<session>/data)
and animates the landmarks in 3D over time, drawing the bones between them.
Two panels are shown: a three-quarter view and a side view turned 90 degrees
about the vertical axis, where travel in depth reads as left-to-right motion.

Writes MP4 through OpenCV, so no ffmpeg binary is required.
"""
import os

import matplotlib
matplotlib.use("Agg")           # headless - must be set before pyplot
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

# Native Stray Scanner RGB dimensions, and the frame size the pipeline works in
NATIVE_W, NATIVE_H = 1920, 1440
PROC_W, PROC_H = 192, 256           # after the 90 deg CW rotation


def bone_color(a, b):
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
            raise RuntimeError(f"Missing coordinate CSV: {path}")
        arr = np.loadtxt(path, delimiter=",")
        if arr.ndim == 1:                    # single-frame session
            arr = arr.reshape(1, 3)
        out[name] = arr
    n = min(len(a) for a in out.values())
    return {k: v[:n] for k, v in out.items()}


def repair_intrinsics(coords, camera_matrix_path):
    """Undo pipelandmark.py's incorrect intrinsics scaling and re-project.

    pipelandmark.py scales the camera matrix with
        scale_x = height / 1440      scale_y = width / 1920
    but both axes shrink by one uniform factor (1920->256 and 1440->192 are both
    1/7.5), and the 90 degree rotation applied to every frame never reaches the
    camera matrix. That puts the principal point at 89% of the width and 28% of
    the height and leaves fx/fy = 1.78 although the source has square pixels.
    The visible symptom is a subject walking toward the camera appearing to rise.

    z comes straight from the depth map and is untouched, so the pixel
    coordinates can be recovered exactly and re-projected correctly.
    """
    M = np.loadtxt(camera_matrix_path, delimiter=",")
    f, cx0, cy0 = M[0, 0], M[0, 2], M[1, 2]

    fx_w, fy_w = f * (PROC_H / NATIVE_H), f * (PROC_W / NATIVE_W)
    cx_w, cy_w = cx0 * (PROC_H / NATIVE_H), cy0 * (PROC_W / NATIVE_W)

    s = PROC_W / NATIVE_H               # == PROC_H / NATIVE_W == 1/7.5
    fx_c = fy_c = f * s
    cx_c = (PROC_W - 1) - cy0 * s       # axes swap under the rotation
    cy_c = cx0 * s

    out = {}
    for name, arr in coords.items():
        x, y, z = arr[:, 0], arr[:, 1], arr[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            px = x * fx_w / z + cx_w
            py = y * fy_w / z + cy_w
            out[name] = np.column_stack([(px - cx_c) * z / fx_c,
                                         (py - cy_c) * z / fy_c, z])
    return out


def _to_plot_space(arr):
    """Camera coords (x right, y down, z depth) -> plot coords (x, depth, up)."""
    return np.column_stack([arr[:, 0], arr[:, 2], -arr[:, 1]])


def render_movie(landmarks, out_path, fps=60, size=(1100, 430), elev=12,
                 azim=-70, side_view=True, progress_every=10):
    """Animate the landmarks in 3D and write an MP4 to out_path."""
    pts = {k: _to_plot_space(v) for k, v in landmarks.items()}
    n_frames = len(next(iter(pts.values())))
    seq = np.stack([pts[name] for name in LANDMARKS], axis=1)

    # The tracker writes (0,0,0) when a joint was never seen; those would drag
    # the framing toward the origin.
    valid = ~np.all(np.isclose(seq, 0.0), axis=2) & np.isfinite(seq).all(axis=2)
    if not valid.any():
        raise RuntimeError("No usable landmark coordinates to animate")
    masked = np.where(valid[..., None], seq, np.nan)

    centres = np.nanmean(masked, axis=1)
    for i in range(n_frames):
        if not np.isfinite(centres[i]).all():
            centres[i] = centres[i - 1] if i else np.zeros(3)
    win = max(1, min(15, n_frames))
    kernel = np.ones(win) / win
    centres = np.stack([
        np.convolve(np.pad(centres[:, a], (win // 2, win - 1 - win // 2),
                           mode="edge"), kernel, mode="valid")
        for a in range(3)
    ], axis=1)
    true_depth = centres[:, 1].copy()

    flat = masked.reshape(-1, 3)
    flat = flat[np.isfinite(flat).all(axis=1)]
    lo, hi = flat.min(axis=0), flat.max(axis=0)

    # Every axis is a true distance from the camera, and the box is fitted to
    # the walk so the subject translates through the scene. Per-axis limits
    # rather than a cube keep the body a usable size over a multi-metre walk;
    # box_aspect restores uniform scale so limbs are not stretched.
    rng = np.maximum(hi - lo, 0.25)
    limits = np.array([lo - 0.06 * rng, hi + 0.06 * rng])
    r = limits[1] - limits[0]
    aspect = tuple(r / r.max())

    panels = [(azim, "Three-quarter view")]
    if side_view:
        panels.append((azim + 90.0, "Side view (+90 deg)"))

    w, h = size
    dpi = 100
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)

    drawables = []
    for k, (pan_azim, pan_title) in enumerate(panels, start=1):
        ax = fig.add_subplot(1, len(panels), k, projection="3d")
        ax.view_init(elev=elev, azim=pan_azim)

        # Set once, never touched again inside the loop, so only points move
        ax.set_xlim(limits[0][0], limits[1][0])
        ax.set_ylim(limits[0][1], limits[1][1])
        ax.set_zlim(limits[0][2], limits[1][2])
        try:
            ax.set_box_aspect(aspect)
        except AttributeError:          # matplotlib < 3.3
            pass

        ax.set_xlabel("X from camera (m)", fontsize=8, labelpad=2)
        ax.set_ylabel("Depth from camera (m)", fontsize=8, labelpad=6)
        # Negative values are expected: the camera axis origin sits above the
        # subject, so this is height relative to the camera, not the floor.
        ax.set_zlabel("Height rel. camera (m)", fontsize=8, labelpad=8)
        ax.set_title(pan_title, fontsize=9, pad=2)

        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_major_locator(
                matplotlib.ticker.MaxNLocator(nbins=5, prune=None))
        if k == 2:
            # X is nearly edge-on in the side view, so its ticks collide
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=3))
        ax.tick_params(labelsize=7)

        # Build the artists once and only update their data each frame
        bone_lines = []
        for a, b in BONES:
            line, = ax.plot([], [], [], "-", lw=2.2, color=bone_color(a, b))
            bone_lines.append((a, b, line))
        joints, = ax.plot([], [], [], "o", ms=4.5, color="#f0f6fc",
                          markeredgecolor="#30363d", linestyle="None")
        drawables.append((bone_lines, joints))

    label = fig.text(0.01, 0.972, "", ha="left", fontsize=9, color="#57606a")
    # right < 1 leaves room for the side panel's axis label, which otherwise
    # runs off the canvas edge
    # Margins leave room for the axis labels. A long walk gives a flat, wide box
    # whose labels sit close to the frame; a session with little depth travel
    # gives a near-cubic box with much taller 3D axes, which needs more room
    # underneath or the x/depth labels run off the canvas.
    tall_box = aspect[2] > 0.5 * max(aspect)
    fig.subplots_adjust(left=0.02, right=0.955, top=0.91,
                        bottom=0.17 if tall_box else 0.08, wspace=0.02)

    index = {name: k for k, name in enumerate(LANDMARKS)}
    writer = None
    try:
        for i in range(n_frames):
            fp = seq[i]
            for bone_lines, joints in drawables:
                for a, b, line in bone_lines:
                    pa, pb = fp[index[a]], fp[index[b]]
                    line.set_data([pa[0], pb[0]], [pa[1], pb[1]])
                    line.set_3d_properties([pa[2], pb[2]])
                joints.set_data(fp[:, 0], fp[:, 1])
                joints.set_3d_properties(fp[:, 2])

            label.set_text(f"frame {i + 1}/{n_frames}    "
                           f"t = {i / float(fps):5.2f}s    "
                           f"depth {true_depth[i]:4.2f} m")

            fig.canvas.draw()
            bgr = cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba()),
                               cv2.COLOR_RGBA2BGR)

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


def main(folder=None, out=None, camera_matrix=None):
    if folder is None:
        folder = input("Folder name: ")
    if out is None:
        out = input("Output file name (.mp4): ")
    data_dir = os.path.join("charts", folder, "data")
    lm = load_landmarks(data_dir)
    if camera_matrix is None:
        guess = os.path.join(folder, "camera_matrix.csv")
        camera_matrix = guess if os.path.exists(guess) else None
    if camera_matrix:
        lm = repair_intrinsics(lm, camera_matrix)
    render_movie(lm, out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
