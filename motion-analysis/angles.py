"""Joint angles and CSV/graph output — shared by both subject kinds.

Which angles get computed comes from the profile: six for a torso (elbows, hips,
knees), fifteen for a hand (three flexion angles per finger). The arithmetic and
the output layout are identical, and match the gait pipeline so anything
downstream is already familiar:

    <out>/data/<test>_<landmark>.csv       time_s, x_m, y_m, z_m — one row per frame
    <out>/data/<test>_<joint>_angle.csv    time_s, angle_deg — one row per frame
    <out>/graphs/<test>_<joint>_angle.png
    <out>/graphs/<test>_<landmark>_distance.png

Naming and column headers come from csvout, which the torso pipeline also uses,
so both produce identically-spelled output.
"""
import os

import numpy as np

import csvout

UNTRACKED = (0, 0, 0)


def calculate_angle(A, B, C):
    """Angle at B, in degrees.

    NaN rather than None for a degenerate triangle: 'None' written into a CSV
    makes the whole column unparseable by loadtxt and pandas.
    """
    A, B, C = np.asarray(A, float), np.asarray(B, float), np.asarray(C, float)
    BA, BC = A - B, C - B
    nBA, nBC = np.linalg.norm(BA), np.linalg.norm(BC)
    if nBA == 0 or nBC == 0:
        return np.nan
    cosine = np.clip(np.dot(BA, BC) / (nBA * nBC), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def compute(arrays, profile):
    """{pivot: [degrees per frame]} from {landmark: [(x, y, z), ...]}."""
    n = len(next(iter(arrays.values())))
    out = {}
    for pivot, (b, a, c) in profile["angles"].items():
        series = []
        for i in range(n):
            # any of the three being untracked makes the angle meaningless
            if (tuple(arrays[b][i]) == UNTRACKED or tuple(arrays[a][i]) == UNTRACKED
                    or tuple(arrays[c][i]) == UNTRACKED):
                series.append(np.nan)
                continue
            series.append(calculate_angle(arrays[a][i], arrays[b][i], arrays[c][i]))
        out[pivot] = series
    return out


def _plot(x, y, title, ylabel, path, color=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 4.5))
    plt.plot(x, y, color=color)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=90)
    plt.close()


def write(arrays, angle_series, profile, out_dir, fps=60.0, graphs=True,
          test_id=None):
    data_dir = os.path.join(out_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    for name in profile["landmarks"]:
        csvout.write_point(data_dir, name, arrays[name], test_id=test_id, fps=fps)
    for pivot, series in angle_series.items():
        csvout.write_angle(data_dir, pivot, series, test_id=test_id, fps=fps)

    if not graphs:
        return
    gdir = os.path.join(out_dir, "graphs")
    os.makedirs(gdir, exist_ok=True)
    for pivot, series in angle_series.items():
        t = np.arange(len(series)) / fps
        _plot(t, series, f"{pivot} angle over time", "Angle (degrees)",
              csvout.graph_path(gdir, pivot, "angle", test_id), color="purple")
    for name in profile["traces"]:
        arr = np.asarray(arrays[name], float)
        z = np.where(np.all(arr == 0, axis=1), np.nan, arr[:, 2])
        t = np.arange(len(z)) / fps
        _plot(t, z, f"Z distance over time ({name})", "Z distance (m)",
              csvout.graph_path(gdir, name, "distance", test_id))


def summary(arrays, angle_series, profile):
    """Short text block: tracking rate, and the range of each angle.

    Printed so a run can be sanity-checked at a glance without opening the CSVs.
    """
    n = len(next(iter(arrays.values())))
    tracked = sum(1 for i in range(n)
                  if tuple(arrays[next(iter(profile['landmarks']))][i]) != UNTRACKED)
    lines = [f"frames {n}, landmarks tracked on {tracked} "
             f"({100.0 * tracked / max(n, 1):.1f}%)",
             f"{'joint':16}{'min':>8}{'median':>9}{'max':>8}   valid"]
    for pivot, series in angle_series.items():
        v = np.asarray(series, float)
        v = v[np.isfinite(v)]
        if not v.size:
            lines.append(f"{pivot:16}{'—':>8}{'—':>9}{'—':>8}      0%")
            continue
        lines.append(f"{pivot:16}{v.min():8.1f}{np.median(v):9.1f}{v.max():8.1f}"
                     f"{100.0 * v.size / max(n, 1):7.0f}%")
    return "\n".join(lines)
