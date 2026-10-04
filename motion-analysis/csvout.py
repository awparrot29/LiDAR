"""File naming and CSV layout, shared by both output paths.

Two separate pipelines write results — gait-analysis/calculateangle.py for a
torso and motion-analysis/angles.py for every other profile. They used to spell
their output independently, so the rules below live here instead and both import
them. Changing a unit or a separator in one place changes it everywhere.

Three rules, all visible in the output:

1. **Names carry the MDS-UPDRS test.** `3.7a_left_knee.csv`, not `left knee.csv`.
   Results from several tests can then be extracted into one folder without
   colliding, and a loose CSV still says what it came from.
2. **Underscores, never spaces.** Spaces in filenames break shell pipelines and
   force quoting in every downstream script.
3. **Every column states its unit in the header row** — `x_m`, `angle_deg`,
   `time_s`. Previously these files had no header at all, so a reader had to
   already know that column 2 was metres.

The header row is new as of 2026-10-04 and is a breaking change for anything
reading the files back: `np.loadtxt` treats it as data unless given
`skiprows=1`. gait-analysis/skeleton3d.load_landmarks is the only in-tree
reader and has been updated. pandas `read_csv` now picks the names up for free,
which it could not do before.
"""
import os
import re

import numpy as np

POINT_HEADER = ("x_m", "y_m", "z_m")
ANGLE_HEADER = ("angle_deg",)
TIME_HEADER = "time_s"

# Coordinates keep %s so the values are byte-identical to what this project
# produced before the header was added; only time gets a fixed width, because
# n/60 in full float repr is 0.016666666666666666 and unreadable in a spreadsheet.
_TIME_FMT = "%.5f"
_VALUE_FMT = "%s"


def safe(name):
    """Filename-safe form of `name`: spaces to underscores, oddities dropped.

    The dot survives so a test id stays `3.7a` rather than `37a`, and the plain
    hyphen survives so `Pronation-Supination` keeps its compound. The en-dash
    the MDS-UPDRS labels use as a separator becomes a space first, so
    `Finger Tapping - Right Hand` lands as `Finger_Tapping_Right_Hand` rather
    than growing a stray double underscore where the dash was stripped.
    """
    s = str(name).strip()
    s = re.sub(r"[‒-―]", " ", s)      # figure/en/em dashes
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^\w.\-]", "", s)
    s = re.sub(r"_{2,}", "_", s).strip("_")
    return s or "unnamed"


def stem(name, test_id=None):
    """`3.7a_left_knee` when the test is known, `left_knee` when it is not.

    test_id is optional throughout: the CLI can be pointed at a session with no
    MDS-UPDRS item in mind, and that should still produce usable output.
    """
    base = safe(name)
    return f"{safe(test_id)}_{base}" if test_id else base


def wants_time(name):
    """True for the fingertip landmarks.

    The tip trajectories are what tremor and tapping analysis read, and there a
    frame index is not enough — the sample rate has to be on the page, because
    the clinically interesting quantity is a frequency. Every other landmark is
    one row per frame at a known fps and does not need the column.
    """
    return str(name).strip().lower().endswith("tip")


def write(path, values, header, fps=60.0, with_time=False):
    """Write one measurement CSV with a unit-bearing header row.

    `values` is (n,) for a scalar series or (n, k) for coordinates.
    """
    arr = np.asarray(values, float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    cols = tuple(header)
    fmt = [_VALUE_FMT] * arr.shape[1]
    if with_time:
        t = (np.arange(len(arr)) / float(fps)).reshape(-1, 1)
        arr = np.hstack([t, arr])
        cols = (TIME_HEADER,) + cols
        fmt = [_TIME_FMT] + fmt

    # comments="" or numpy prefixes the header line with "# ", which turns the
    # first column name into "# x_m" for every reader that is not numpy.
    np.savetxt(path, arr, delimiter=",", header=",".join(cols),
               comments="", fmt=fmt)
    return path


def write_point(data_dir, name, coords, test_id=None, fps=60.0):
    """One landmark's x/y/z in metres, one row per frame."""
    path = os.path.join(data_dir, stem(name, test_id) + ".csv")
    return write(path, coords, POINT_HEADER, fps=fps, with_time=wants_time(name))


def write_angle(data_dir, name, series, test_id=None, fps=60.0):
    """One joint's angle in degrees, one row per frame."""
    path = os.path.join(data_dir, stem(f"{name} angle", test_id) + ".csv")
    return write(path, series, ANGLE_HEADER, fps=fps, with_time=wants_time(name))


def graph_path(graph_dir, name, suffix, test_id=None):
    """Matching path for a PNG trace, named by the same rules."""
    return os.path.join(graph_dir, stem(f"{name} {suffix}", test_id) + ".png")
