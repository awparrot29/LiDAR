"""Regenerate clean output for every example, with both pipelines.

For each example session this writes, next to the recording:

    <example>/output/lidar/     data/*.csv  graphs/*.png   landmarks.npz
    <example>/output/nolidar/   data/*.csv  graphs/*.png   landmarks.npz

Same 12 joint-coordinate CSVs and 6 angle CSVs the existing pipeline produces, so
the two are directly comparable and nothing downstream has to change. The
landmarks.npz is kept so compare.py can score the run without re-tracking.

Usage:
    python run_examples.py                    # every example under ../Examples
    python run_examples.py <folder> [...]     # specific sessions
    python run_examples.py --only nolidar     # skip the LiDAR pipeline
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LIDAR_DIR = os.path.normpath(os.path.join(HERE, "..", "LiDAR-Gait-Analysis",
                                          "gait-analysis"))
EXAMPLES = os.path.normpath(os.path.join(HERE, "..", "Examples"))

# Same pivot -> (pivot, a, c) map the LiDAR pipeline's calculateangle.py uses
ANGLE_MAP = {
    "left elbow": ("left elbow", "left shoulder", "left wrist"),
    "right elbow": ("right elbow", "right shoulder", "right wrist"),
    "left hip": ("left hip", "left shoulder", "left knee"),
    "right hip": ("right hip", "right shoulder", "right knee"),
    "left knee": ("left knee", "left hip", "left ankle"),
    "right knee": ("right knee", "right hip", "right ankle"),
}


def calculate_angle(A, B, C):
    """Angle at B in degrees; NaN when a vector collapses."""
    BA = np.asarray(A, float) - np.asarray(B, float)
    BC = np.asarray(C, float) - np.asarray(B, float)
    nBA, nBC = np.linalg.norm(BA), np.linalg.norm(BC)
    if nBA == 0 or nBC == 0:
        return np.nan
    return float(np.degrees(np.arccos(
        np.clip(np.dot(BA, BC) / (nBA * nBC), -1.0, 1.0))))


def track(folder, which):
    """Run one pipeline over a session -> {landmark name: [(x, y, z), ...]}."""
    if which == "lidar":
        if LIDAR_DIR not in sys.path:
            sys.path.insert(0, LIDAR_DIR)
        import pipelandmark
        return pipelandmark.extract_all_landmarks(folder)
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    import poselandmark
    return poselandmark.extract_all_landmarks(folder)


def write_output(arrays, out_dir, graphs=True):
    data_dir = os.path.join(out_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    for name, pts in arrays.items():
        np.savetxt(os.path.join(data_dir, f"{name}.csv"),
                   np.asarray(pts, float), delimiter=",", fmt="%s")
    angles = {}
    for pivot, (b, a, c) in ANGLE_MAP.items():
        series = [calculate_angle(arrays[a][i], arrays[b][i], arrays[c][i])
                  for i in range(len(arrays[b]))]
        angles[pivot] = series
        np.savetxt(os.path.join(data_dir, f"{pivot} angle.csv"),
                   np.asarray(series, float), delimiter=",", fmt="%s")
    np.savez(os.path.join(out_dir, "landmarks.npz"),
             **{k: np.asarray(v, float) for k, v in arrays.items()})

    if graphs:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        gdir = os.path.join(out_dir, "graphs")
        os.makedirs(gdir, exist_ok=True)
        for pivot, series in angles.items():
            t = np.arange(len(series)) / 60.0
            plt.figure(figsize=(10, 4))
            plt.plot(t, series, color="purple")
            plt.title(f"{pivot.capitalize()} Angle Over Time")
            plt.xlabel("Time (s)")
            plt.ylabel("Angle (degrees)")
            plt.grid(True, alpha=.3)
            plt.tight_layout()
            plt.savefig(os.path.join(gdir, f"{pivot} angle.png"), dpi=90)
            plt.close()
        for name in ("left knee", "right knee", "left wrist", "right wrist"):
            arr = np.asarray(arrays[name], float)
            t = np.arange(len(arr)) / 60.0
            plt.figure(figsize=(10, 4))
            plt.plot(t, arr[:, 2])
            plt.title(f"Z Distance Over Time ({name})")
            plt.xlabel("Time (s)")
            plt.ylabel("Z Distance (m)")
            plt.grid(True, alpha=.3)
            plt.tight_layout()
            plt.savefig(os.path.join(gdir, f"{name} distance.png"), dpi=90)
            plt.close()
    return angles


def find_examples():
    out = []
    for kind in ("torso", "hand"):
        d = os.path.join(EXAMPLES, kind)
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            p = os.path.join(d, name)
            if os.path.isdir(p) and os.path.exists(os.path.join(p, "rgb.mp4")):
                out.append(p)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="*")
    ap.add_argument("--only", choices=("lidar", "nolidar"), default=None)
    ap.add_argument("--no-graphs", action="store_true")
    args = ap.parse_args(argv)

    folders = args.folders or find_examples()
    if not folders:
        raise SystemExit(f"No examples found under {EXAMPLES}")
    which = [args.only] if args.only else ["lidar", "nolidar"]

    for folder in folders:
        label = os.path.basename(os.path.normpath(folder))
        for w in which:
            out_dir = os.path.join(folder, "output", w)
            t0 = time.time()
            print(f"\n===== {label}  [{w}] =====", flush=True)
            try:
                arrays = track(folder, w)
            except Exception as exc:            # a hand session has no legs to track
                print(f"  FAILED ({type(exc).__name__}: {exc})")
                continue
            os.makedirs(out_dir, exist_ok=True)
            write_output(arrays, out_dir, graphs=not args.no_graphs)
            n = len(next(iter(arrays.values())))
            print(f"  wrote {out_dir}  ({n} frames, {time.time()-t0:.0f}s)",
                  flush=True)
    print("\nDone.")


if __name__ == "__main__":
    main()
