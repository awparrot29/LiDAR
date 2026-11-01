#!/usr/bin/env python3
"""LiDAR Parkinson's movement analysis - offline command-line version.

Does what the web app does, with no web server and no internet connection:
give it a Stray Scanner recording and the MDS-UPDRS Part 3 test it was
recorded for, and it writes a ZIP of joint-coordinate and joint-angle CSVs,
the 3D skeleton / camera / LiDAR movies, and a README explaining every file.

Every option on the website is a command-line argument here:

    website                       command line
    -------                       ------------
    choose recording (.zip)       INPUT  (a .zip, or an unzipped session folder)
    MDS-UPDRS Part 3 Test menu    --test 3.10
    Tracker (MediaPipe/RTMPose)   --tracker mediapipe | rtmpose
    Download results              --output results.zip

Examples:

    python lidar_cli.py --list-tests
    python lidar_cli.py recording.zip --test 3.10
    python lidar_cli.py recording.zip --test 3.4a -o patient07_3.4a.zip
    python lidar_cli.py session_folder --test 3.10 --tracker rtmpose

Offline use (hospital / firewalled networks)
--------------------------------------------
MediaPipe's models ship inside its pip package, so hand, gait, rest-tremor and
auto-detect runs need nothing else. The RTMPose models (tests 3.7, 3.8 and the
--tracker rtmpose option) and the SpinePose model (test 3.13) are normally
downloaded on first use. Behind a firewall that download fails, so instead:

    1. On any computer WITH internet access, in this repository:
           python lidar_cli.py --download-models
       This fills the models/ folder (~0.6 GB).
    2. Copy that models/ folder into the repository on the offline computer.

lidar_cli.py always loads models from models/ and never touches the network.
It checks for the files it needs before starting, so a missing model is
reported straight away rather than as a network error partway through a run.
"""
import argparse
import collections
import csv
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
GAIT_DIR = os.path.join(HERE, "gait-analysis")
MOTION_DIR = os.path.join(HERE, "motion-analysis")
DEFAULT_MODELS_DIR = os.path.join(HERE, "models")
DEFAULT_TIMEOUT = 3000  # seconds, same as the website
FPS = 60.0

# Working filenames inside the temp folder; the files in the output ZIP are
# renamed after the test, exactly as the website names its downloads.
MOVIE_NAME = "gait_skeleton_3d.mp4"
RGB_NAME = "preview_rgb.mp4"
DEPTH_NAME = "preview_lidar.mp4"

# ---------------------------------------------------------------------------
# MDS-UPDRS Part 3 tests -> pipeline. Copied from app.py (TEST_TO_KIND and
# TEST_LABELS) rather than imported, because importing app.py starts Flask and
# creates the server's /home folders. Change both together.
# ---------------------------------------------------------------------------
TESTS = {
    "3.4a": ("Finger Tapping – Right Hand", "hand"),
    "3.4b": ("Finger Tapping – Left Hand", "hand"),
    "3.5a": ("Hand Movements – Right Hand", "hand"),
    "3.5b": ("Hand Movements – Left Hand", "hand"),
    "3.6a": ("Pronation-Supination – Right Hand", "hand"),
    "3.6b": ("Pronation-Supination – Left Hand", "hand"),
    "3.7a": ("Toe Tapping – Right Foot", "toe_tap"),
    "3.7b": ("Toe Tapping – Left Foot", "toe_tap"),
    "3.8a": ("Leg Agility – Right Leg", "foot"),
    "3.8b": ("Leg Agility – Left Leg", "foot"),
    "3.9": ("Arising from Chair", "torso"),
    "3.10": ("Gait", "torso"),
    "3.11": ("Freezing of Gait", "torso"),
    "3.12": ("Postural Stability", "torso"),
    "3.13": ("Posture", "spine"),
    "3.15a": ("Postural Tremor – Right Hand", "hand"),
    "3.15b": ("Postural Tremor – Left Hand", "hand"),
    "3.16a": ("Kinetic Tremor – Right Hand", "hand"),
    "3.16b": ("Kinetic Tremor – Left Hand", "hand"),
    "3.17": ("Rest Tremor – All Limbs", "rest_tremor"),
}
KINDS = ("auto", "torso", "hand", "foot", "toe_tap", "rest_tremor", "spine")
MOTION_KINDS = ("hand", "foot", "toe_tap", "rest_tremor", "spine")
TRACKERS = ("mediapipe", "rtmpose")

KIND_INFO = {
    "torso": "Whole body, 12 joints (shoulders, elbows, wrists, hips, knees, "
             "ankles) and 6 joint angles.",
    "hand": "One hand, 21 landmarks (wrist plus 4 per finger) and 15 finger "
            "flexion angles.",
    "foot": "Whole body plus feet, 18 landmarks (body joints plus heel, big toe "
            "and small toe on each foot) and 8 angles including ankle "
            "dorsiflexion.",
    "toe_tap": "Lower legs only, 10 landmarks (knees, ankles, heels, big and "
               "small toes) and both ankle dorsiflexion angles.",
    "rest_tremor": "Whole body, same 12 joints and 6 angles as the gait "
                   "pipeline, with light smoothing so 4-6 Hz tremor is kept.",
    "spine": "Spine and posture, 14 landmarks (5 vertebral levels, neck, head, "
             "hips, shoulders, knees) and 6 spinal segment angles.",
}

# What each test's numbers are most useful for. Factual pointers to the
# relevant files only - scoring is left to the clinician.
TEST_NOTES = {
    "hand_tapping": "For finger tapping, the distance between the thumb_tip and "
                    "index_tip coordinates gives the tap amplitude over time; "
                    "its peaks give the tap rate and any decrement.",
    "hand_movements": "For hand opening/closing, the finger flexion angles "
                      "(MCP/PIP) rise and fall with each movement.",
    "pronation": "For pronation-supination, compare the thumb_tip and pinky_tip "
                 "positions relative to the wrist: they swap sides as the hand "
                 "rotates.",
    "postural_tremor": "For tremor, the fingertip and wrist coordinates are "
                       "smoothed lightly (3 frames, about 50 ms) so 4-8 Hz "
                       "tremor is preserved. A frequency analysis of these "
                       "traces shows tremor frequency and amplitude.",
    "toe_tap": "For toe tapping, the big_toe and heel coordinates and the ankle "
               "dorsiflexion files show each tap. About 90 degrees is a neutral "
               "ankle; a smaller angle means the toes are lifted toward the "
               "shin.",
    "leg_agility": "For leg agility, the knee and ankle coordinates show how "
                   "high each stomp lifts; the hip and knee angles show the "
                   "range of motion.",
    "gait": "For gait and chair rise, the knee and hip angles and the ankle "
            "and wrist coordinates show stride, arm swing and range of motion. "
            "Recordings are of forward/backward walks toward the camera.",
    "posture": "For posture, each spinal angle is 180 degrees when that section "
               "is straight; a smaller angle means that section is more bent. "
               "The skeleton video shows which way (usually a forward stoop).",
    "rest_tremor": "Rest tremor (3.17) is one recording of all four limbs: use "
                   "the wrist files for the right/left upper extremity (RUE/LUE) "
                   "and the ankle files for the right/left lower extremity "
                   "(RLE/LLE). Smoothing is light (3 frames) so 4-6 Hz tremor "
                   "is preserved.",
}
TEST_NOTE_KEY = {
    "3.4a": "hand_tapping", "3.4b": "hand_tapping",
    "3.5a": "hand_movements", "3.5b": "hand_movements",
    "3.6a": "pronation", "3.6b": "pronation",
    "3.7a": "toe_tap", "3.7b": "toe_tap",
    "3.8a": "leg_agility", "3.8b": "leg_agility",
    "3.9": "gait", "3.10": "gait", "3.11": "gait", "3.12": "gait",
    "3.13": "posture",
    "3.15a": "postural_tremor", "3.15b": "postural_tremor",
    "3.16a": "postural_tremor", "3.16b": "postural_tremor",
    "3.17": "rest_tremor",
}

# ---------------------------------------------------------------------------
# Model weights that are downloaded on first use rather than shipped in a pip
# package. Filenames are what rtmlib / spinepose save them as.
# ---------------------------------------------------------------------------
_YOLOX_M = "yolox_m_8xb8-300e_humanart-c2c7a14a.onnx"
MODEL_FILES = {
    # rtmlib Body(mode="balanced"): person detector + RTMPose-m
    "rtmpose_body": ("rtmlib", [
        _YOLOX_M,
        "rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.onnx"]),
    # rtmlib Wholebody(mode="balanced"): person detector + RTMW whole body
    "rtmpose_wholebody": ("rtmlib", [
        _YOLOX_M,
        "rtmw-dw-x-l_simcc-cocktail14_270e-256x192_20231122.onnx"]),
    # spinepose PoseTracker(mode="large"): RF-DETR detector + SpinePose-l
    "spinepose": ("spinepose", [
        "rfdetr_m_v142_576x576.onnx",
        "spinepose-l_32xb256-10e_simspine-256x192.onnx"]),
}


def models_needed(kind, tracker):
    """Keys of MODEL_FILES a run needs. MediaPipe needs nothing downloaded."""
    if kind in ("foot", "toe_tap"):
        return ["rtmpose_wholebody"]
    if kind == "spine":
        return ["spinepose"]
    if kind == "torso" and tracker == "rtmpose":
        return ["rtmpose_body"]
    return []


def checkpoint_dir(models_dir, lib):
    """Where a library's weights live under models/ (mirrors its own cache)."""
    return os.path.join(models_dir, lib, "hub", "checkpoints")


def missing_models(models_dir, keys):
    out = []
    for key in keys:
        lib, files = MODEL_FILES[key]
        for f in files:
            if not os.path.isfile(os.path.join(checkpoint_dir(models_dir, lib), f)):
                out.append(os.path.join(lib, "hub", "checkpoints", f))
    return out


def child_env(models_dir):
    """Environment for the pipeline process: rtmlib reads its cache from
    TORCH_HOME, so pointing it at models/rtmlib keeps it offline."""
    env = dict(os.environ)
    env["TORCH_HOME"] = os.path.join(models_dir, "rtmlib")
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def spinepose_redirect(models_dir):
    """Statement that points spinepose at models/spinepose instead of
    ~/.cache/spinepose. spinepose has no environment variable for this, but it
    looks the folder up through this function every time it loads a model."""
    hub = os.path.join(models_dir, "spinepose", "hub")
    return ("import spinepose.tools.utils.file as _spf; "
            f"_spf._get_cache_dir = lambda: {hub!r}")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def safe(name):
    """Filename-safe text. Same rules as motion-analysis/csvout.safe()."""
    import re
    s = str(name).strip()
    s = re.sub(r"[‒-―]", " ", s)      # figure/en/em dashes
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^\w.\-]", "", s)
    s = re.sub(r"_{2,}", "_", s).strip("_")
    return s or "unnamed"


def label_of(test_id):
    """Test label with a plain hyphen, for consoles and README text."""
    return TESTS[test_id][0].replace("–", "-")


def output_names(test_id):
    """(zip, skeleton, rgb, lidar) names, same as the website's downloads."""
    base = safe(f"{test_id} {TESTS[test_id][0]}") if test_id else "results"
    return (f"{base}.zip", f"{base}_skeleton.mp4",
            f"{base}_rgb.mp4", f"{base}_lidar.mp4")


def extract_zip(zip_path, dest_root):
    """Extract, tolerating Windows backslash entry names and blocking `..`."""
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            name = info.filename.replace("\\", "/")
            if name.endswith("/"):
                continue
            parts = [p for p in name.split("/") if p not in ("", ".", "..")]
            if not parts:
                continue
            dest = os.path.join(dest_root, *parts)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with zf.open(info) as src, open(dest, "wb") as dst:
                shutil.copyfileobj(src, dst)


def find_session(root):
    """Relative path of the first folder holding rgb.mp4, camera_matrix.csv
    and depth/ - a Stray Scanner session - or None."""
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d != "__MACOSX"]
        if "rgb.mp4" in files and "camera_matrix.csv" in files and "depth" in dirs:
            return os.path.relpath(dirpath, root)
    return None


def git_version():
    try:
        out = subprocess.run(["git", "-C", HERE, "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


class Progress:
    """One updating line on a terminal; a line per 10% when output is a file."""

    def __init__(self, quiet=False):
        self.quiet = quiet
        self.tty = sys.stderr.isatty()
        self.last_bucket = -1
        self.width = 0

    def __call__(self, pct, stage):
        if self.quiet:
            return
        line = f"[{pct:5.1f}%] {stage}"
        if self.tty:
            pad = max(0, self.width - len(line))
            sys.stderr.write("\r" + line + " " * pad)
            sys.stderr.flush()
            self.width = len(line)
        else:
            bucket = int(pct // 10)
            if bucket != self.last_bucket:
                self.last_bucket = bucket
                print(line, file=sys.stderr, flush=True)

    def done(self):
        if self.tty and not self.quiet and self.width:
            sys.stderr.write("\n")
            sys.stderr.flush()
            self.width = 0


def log(msg, quiet=False):
    if not quiet:
        print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Pipeline (same scripts the website's _run_job builds)
# ---------------------------------------------------------------------------
def preview_src(session):
    return "; ".join([
        f"sys.path.insert(0, {MOTION_DIR!r})",
        "import preview",
        f"preview.render_both({session!r}, {RGB_NAME!r}, {DEPTH_NAME!r}, fps=60.0)",
    ])


def detect_kind(work, session, models_dir):
    """Website's auto-detect: torso (walking) vs hand (close-up)."""
    src = "; ".join([
        "import sys, os",
        f"sys.path.insert(0, {MOTION_DIR!r})",
        "import detect",
        f"k, ev = detect.detect(os.path.join({work!r}, {session!r}))",
        'print("@@KIND " + k)',
        "print(detect.explain(ev))",
    ])
    pr = subprocess.run([sys.executable, "-c", src], capture_output=True,
                        text=True, cwd=work, env=child_env(models_dir))
    kind, explain = "torso", ""
    for ln in (pr.stdout or "").splitlines():
        if ln.startswith("@@KIND "):
            kind = ln[7:].strip()
        elif ln.strip():
            explain = ln.strip()
    if kind not in ("torso", "hand"):
        kind = "torso"
    return kind, explain


def build_script(kind, tracker, session, test_id, models_dir, videos):
    data_rel = os.path.join("charts", session, "data")
    if kind in MOTION_KINDS:
        out_rel = os.path.join("charts", session)
        lines = ["import sys, os",
                 'import matplotlib; matplotlib.use("Agg")']
        if kind == "spine":
            lines.append(spinepose_redirect(models_dir))
        lines += [
            f"sys.path.insert(0, {MOTION_DIR!r})",
            "import extract, angles, profiles, skeleton3d",
            f"_p = profiles.get({kind!r})",
            f"_a, _geom = extract.extract_all_landmarks({session!r}, kind={kind!r})",
            "_ang = angles.compute(_a, _p)",
            f"angles.write(_a, _ang, _p, {out_rel!r}, fps=60.0, graphs=True, "
            f"test_id={test_id or None!r})",
        ]
        if videos:
            lines += [f"skeleton3d.render(_a, _p, {MOVIE_NAME!r}, fps=60.0)",
                      preview_src(session)]
        return "; ".join(lines)

    lines = ["import sys, os",
             'import matplotlib; matplotlib.use("Agg")',
             f"sys.path.insert(0, {GAIT_DIR!r})"]
    if tracker == "rtmpose":
        lines += ["import pipelandmark_rtmpose as _t",
                  'sys.modules["pipelandmark"] = _t']
    lines += ["import calculateangle",
              f"calculateangle.main(folder={session!r}, test_id={test_id or None!r})"]
    if videos:
        lines += ["import skeleton3d",
                  f"_lm = skeleton3d.load_landmarks({data_rel!r}, "
                  f"test_id={test_id or None!r})",
                  f"skeleton3d.render_movie(_lm, {MOVIE_NAME!r})",
                  preview_src(session)]
    return "; ".join(lines)


def run_pipeline(script, work, models_dir, timeout, progress, verbose):
    """Run the pipeline child, turning its @@ markers into progress, exactly
    as the website's progress bar does. Returns the last 100 output lines."""
    proc = subprocess.Popen([sys.executable, "-u", "-c", script], cwd=work,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace",
                            bufsize=1, env=child_env(models_dir))
    tail = collections.deque(maxlen=100)
    deadline = time.monotonic() + timeout
    TRACK, ANGLE, RENDER, MID = 55, 60, 90, 94
    tracked = False
    for raw in proc.stdout:
        tail.append(raw)
        line = raw.strip()
        if verbose:
            progress.done()
            print("  | " + line, file=sys.stderr, flush=True)
        try:
            if line.startswith("@@FRAME "):
                done, total = (int(v) for v in line[8:].split("/"))
                if total:
                    tracked = True
                    progress(TRACK * done / total,
                             f"Tracking pose - frame {done}/{total}")
            elif line.startswith("@@JOINT "):
                frac, name = line[8:].split(" ", 1)
                cur, n = (int(v) for v in frac.split("/"))
                if tracked:
                    progress(TRACK + (ANGLE - TRACK) * cur / n,
                             f"Computing angles - {name} ({cur}/{n})")
                else:
                    progress(TRACK * (cur - 1) / n,
                             f"Loading cached joint {cur}/{n} - {name}")
            elif line.startswith("@@RENDER "):
                done, total = (int(v) for v in line[9:].split("/"))
                if total:
                    progress(ANGLE + (RENDER - ANGLE) * done / total,
                             f"Rendering 3D movie - frame {done}/{total}")
            elif line.startswith("@@PREVIEW "):
                label, frac = line[10:].split(" ", 1)
                done, total = (int(v) for v in frac.split("/"))
                lo, hi = (RENDER, MID) if label == "rgb" else (MID, 98)
                if total:
                    progress(lo + (hi - lo) * done / total,
                             f"Rendering {label} movie - frame {done}/{total}")
        except ValueError:
            pass
        if time.monotonic() > deadline:
            proc.kill()
            raise RuntimeError(f"Pipeline timed out after {timeout // 60} minutes "
                               "(raise the limit with --timeout).")
    if proc.wait() != 0:
        progress.done()
        raise RuntimeError("Pipeline failed. Last output:\n" + "".join(tail)[-3000:])
    return tail


# ---------------------------------------------------------------------------
# README
# ---------------------------------------------------------------------------
def profile_tables(kind):
    """(landmark names, {angle name: (pivot, a, c)}) for a pipeline kind."""
    sys.path.insert(0, MOTION_DIR)
    try:
        import profiles  # imports mediapipe; fine, it is installed for the run
        p = profiles.get(kind)
        return list(p["landmarks"]), dict(p["angles"]), p
    finally:
        sys.path.remove(MOTION_DIR)


def csv_stats(path):
    """(rows, {column: fraction of rows with a number})."""
    with open(path, newline="", encoding="utf-8") as fh:
        rd = csv.reader(fh)
        header = next(rd, [])
        rows, good = 0, collections.Counter()
        for row in rd:
            rows += 1
            for col, val in zip(header, row):
                v = val.strip().lower()
                if v and v not in ("nan", "none", "inf", "-inf"):
                    good[col] += 1
    return rows, header, {c: (good[c] / rows if rows else 0.0) for c in header}


def describe_csv(fname, test_id, landmarks, angle_defs):
    """Plain-language description of one CSV from its name."""
    stem = fname[:-4]
    prefix = safe(test_id) + "_" if test_id else ""
    if prefix and stem.startswith(prefix):
        stem = stem[len(prefix):]
    by_safe_lm = {safe(n): n for n in landmarks}
    by_safe_ang = {safe(f"{n} angle"): n for n in angle_defs}
    if stem in by_safe_ang:
        name = by_safe_ang[stem]
        pivot, a, c = angle_defs[name]
        title = name[0].upper() + name[1:]
        if not title.endswith("flexion"):
            title += " angle"
        if "dorsiflexion" in name:
            scale = ("About 90 = neutral ankle; smaller = toes lifted toward "
                     "the shin; larger = toes pointed down.")
        else:
            scale = ("180 = straight; smaller = more bent. The angle has no "
                     "direction, so it does not say which way the joint bends.")
        return ("angle", name,
                f"{title}, in degrees: the 3D angle at the {pivot} between "
                f"the lines to the {a} and to the {c}. {scale}")
    if stem in by_safe_lm:
        name = by_safe_lm[stem]
        return ("point", name, f"3D position of the {name}, in metres.")
    if stem.endswith("_angle"):
        name = stem[:-6].replace("_", " ")
        return ("angle", name, f"Angle of the {name}, in degrees.")
    name = stem.replace("_", " ")
    return ("point", name, f"3D position of the {name}, in metres.")


def build_readme(info, csv_files, videos, graphs):
    """README.txt text for the output ZIP. `csv_files` is [(name, path)]."""
    kind, test_id = info["kind"], info["test_id"]
    try:
        landmarks, angle_defs, prof = profile_tables(kind)
    except Exception:
        landmarks, angle_defs, prof = [], {}, {}

    L = []
    w = L.append
    title = "LiDAR Movement Analysis - Results"
    w(title)
    w("=" * len(title))
    w("")
    if test_id:
        w(f"Test:         MDS-UPDRS Part 3, item {test_id} - {label_of(test_id)}")
    else:
        w("Test:         (none chosen)")
    w(f"Pipeline:     {kind}{' (auto-detected)' if info['auto'] else ''} - "
      f"{KIND_INFO.get(kind, '')}")
    w(f"Pose model:   {info['model_desc']}")
    w(f"Recording:    {info['input_name']}")
    w(f"Frames:       {info['frames']} at {FPS:g} frames per second "
      f"({info['frames'] / FPS:.1f} s)")
    w(f"Processed:    {info['when']} on {platform.node() or 'this computer'}")
    w(f"Software:     lidar_cli.py, code version {info['version']}, "
      f"Python {platform.python_version()}")
    w(f"Command:      {info['command']}")
    w("")
    w("These are measurements, not a score. They are meant to support, not "
      "replace, the clinician's MDS-UPDRS rating.")
    w("")

    w("HOW THE DATA WAS MADE")
    w("---------------------")
    w("The recording is an iPhone/iPad LiDAR capture made with the Stray "
      "Scanner app: a colour video plus a depth (distance) image for every "
      "frame. For each frame:")
    w("  1. The pose model finds each joint's position in the colour image.")
    w("  2. The LiDAR depth at that spot gives the joint's distance from the "
      "camera.")
    w("  3. Depth readings the LiDAR marked as low-confidence, and readings "
      "that are actually the wall or floor behind the person, are rejected.")
    w("  4. Each joint's depth is averaged over 9 neighbouring frames, dropping "
      "readings more than 2 standard deviations away.")
    win = prof.get("pixel_smooth_window")
    if win:
        w(f"  5. Each joint's image position is smoothed over {win} frames "
          f"({win * 1000 / FPS:.0f} ms) to remove tracker jitter.")
    w("  The image position and depth are then combined into a 3D point "
      "using the camera's lens calibration.")
    w("")

    w("FILE FORMAT")
    w("-----------")
    w("All CSV files are plain comma-separated text with one header row and "
      "one row per video frame. They open directly in Excel, R, MATLAB, SPSS "
      "or Python.")
    w("")
    w("  Coordinate files:  time_s, x_m, y_m, z_m")
    w("  Angle files:       time_s, angle_deg")
    w("")
    w("  time_s   Seconds since the start of the recording (frame / 60).")
    w("  x_m      Metres to the RIGHT of the camera's centre line (as seen in "
      "the upright video).")
    w("  y_m      Metres BELOW the camera's centre line. Note: y increases "
      "downward.")
    w("  z_m      Metres AWAY from the camera (the LiDAR distance).")
    w("  angle_deg  Joint angle in degrees, computed from the 3D points.")
    w("")
    w("  The origin (0, 0, 0) is the camera itself. The camera is assumed to be "
      "still (on a tripod), so positions are relative to the room.")
    w("  'nan' means no value for that frame: the joint was not found, was "
      "hidden, or its depth reading was rejected.")
    w("  Left/right are the person's own left and right, as labelled by the "
      "pose model.")
    w("  LiDAR distance is accurate to about 1-2%.")
    w("")

    if test_id and test_id in TEST_NOTE_KEY:
        w("WHICH FILES TO LOOK AT FOR THIS TEST")
        w("------------------------------------")
        w(TEST_NOTES[TEST_NOTE_KEY[test_id]])
        w("")

    w("FILES IN THIS ZIP")
    w("-----------------")
    w("Valid = share of frames with a usable value.")
    w("")
    w("README.txt")
    w("    This file.")
    w("")
    angles_, points = [], []
    for fname, path in csv_files:
        k, name, desc = describe_csv(fname, test_id, landmarks, angle_defs)
        rows, header, frac = csv_stats(path)
        vcols = [c for c in header if c != "time_s"]
        valid = min((frac[c] for c in vcols), default=0.0)
        (angles_ if k == "angle" else points).append(
            (fname, desc, rows, valid))
    if angles_:
        w("Joint angles (time_s, angle_deg)")
        w("")
        for fname, desc, rows, valid in angles_:
            w(fname)
            w(f"    {desc}")
            w(f"    {rows} rows, valid {valid:.0%}")
        w("")
    if points:
        w("Joint positions (time_s, x_m, y_m, z_m)")
        w("")
        for fname, desc, rows, valid in points:
            w(fname)
            w(f"    {desc}")
            w(f"    {rows} rows, valid {valid:.0%}")
        w("")
    if videos:
        w("Videos (MP4, H.264)")
        w("")
        desc = {
            "skeleton": "3D stick figure drawn from the coordinate CSVs above. "
                        "Use it to check the tracking looks right.",
            "rgb": "The original colour video. PRIVACY: this shows the "
                   "participant and their face; handle it as identifiable "
                   "patient data.",
            "lidar": "The LiDAR depth video (near = warm colours, far = cool). "
                     "Use it to check the depth data was good.",
        }
        for role, fname in videos:
            w(fname)
            w(f"    {desc[role]}")
        w("")
    if graphs:
        w("Graphs (PNG, in the graphs/ folder)")
        w("")
        w("    Quick-look plots of each angle over time ('_angle.png') and of "
          "selected joints' distance from the camera over time "
          "('_distance.png'). They are drawn from the same data as the CSVs.")
        w("")
    w("LOADING THE DATA")
    w("----------------")
    w("Python:  import pandas as pd")
    w("         df = pd.read_csv('<file>.csv')     # columns are named")
    w("R:       df <- read.csv('<file>.csv')")
    w("Excel:   open the .csv file directly.")
    w("")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
def list_tests():
    print("MDS-UPDRS Part 3 tests (use with --test):\n")
    for tid, (_, kind) in TESTS.items():
        print(f"  {tid:<6} {label_of(tid):<36} pipeline: {kind}")
    print("\nNot available (cannot be measured from a skeleton): 3.1 speech, "
          "3.2 facial expression,\n3.3 rigidity, 3.14 global spontaneity, "
          "3.17e lip/jaw tremor, 3.18 constancy of rest tremor.")
    print("\nWithout --test, use --pipeline to choose a pipeline directly "
          f"({', '.join(KINDS)}).")


def download_models(models_dir):
    """Fill models/ so analysis can run with no internet. Run this on a
    machine that has internet access, then copy models/ across."""
    print(f"Model folder: {models_dir}")
    # 1. Reuse anything this machine has already downloaded.
    defaults = {"rtmlib": os.path.expanduser("~/.cache/rtmlib/hub/checkpoints"),
                "spinepose": os.path.expanduser("~/.cache/spinepose/hub/checkpoints")}
    for key, (lib, files) in MODEL_FILES.items():
        dst_dir = checkpoint_dir(models_dir, lib)
        os.makedirs(dst_dir, exist_ok=True)
        for f in files:
            src, dst = os.path.join(defaults[lib], f), os.path.join(dst_dir, f)
            if not os.path.isfile(dst) and os.path.isfile(src):
                print(f"  copying {f} from {defaults[lib]}")
                shutil.copy2(src, dst)

    # 2. Download the rest by loading each model the way the pipelines do.
    if missing_models(models_dir, MODEL_FILES):
        print("Downloading models (this needs internet access)...")
        src = "; ".join([
            spinepose_redirect(models_dir),
            "from rtmlib import Body, Wholebody, PoseTracker",
            "PoseTracker(Body, mode='balanced', backend='onnxruntime', device='cpu')",
            "PoseTracker(Wholebody, mode='balanced', backend='onnxruntime', device='cpu')",
            "from spinepose import PoseTracker as SPT, SpinePoseEstimator",
            "SPT(SpinePoseEstimator, mode='large')",
        ])
        r = subprocess.run([sys.executable, "-c", src], env=child_env(models_dir))
        if r.returncode != 0:
            print("Download failed (see the error above). Is this computer "
                  "online?", file=sys.stderr)
            return 1

    missing = missing_models(models_dir, MODEL_FILES)
    if missing:
        print("These model files are still missing:", file=sys.stderr)
        for m in missing:
            print("  " + m, file=sys.stderr)
        return 1
    total = 0
    for _, (lib, files) in MODEL_FILES.items():
        for f in files:
            total += os.path.getsize(os.path.join(checkpoint_dir(models_dir, lib), f))
    print(f"All models present ({total / 1e6:.0f} MB in {models_dir}).")
    print("Copy this models/ folder into the repository on the offline computer.")
    return 0


def analyse(args):
    q = args.quiet
    t0 = time.monotonic()
    src = os.path.abspath(args.input)
    if not os.path.exists(src):
        log(f"Error: {args.input} does not exist.")
        return 2

    test_id = args.test or ""
    if test_id:
        kind = TESTS[test_id][1]
    else:
        kind = args.pipeline or "auto"

    # Tracker choice only affects the gait/torso pipeline, as on the website.
    if args.tracker == "rtmpose" and kind not in ("torso", "auto"):
        log(f"Note: --tracker only applies to the gait pipeline; the {kind} "
            "pipeline uses its own model.", q)

    zip_name, skel_name, rgb_name, depth_name = output_names(test_id or None)
    out_path = os.path.abspath(args.output or zip_name)
    if os.path.isdir(out_path):
        out_path = os.path.join(out_path, zip_name)
    if not out_path.lower().endswith(".zip"):
        out_path += ".zip"
    if os.path.exists(out_path) and not args.overwrite:
        log(f"Error: {out_path} already exists. Use --overwrite to replace it.")
        return 2

    models_dir = os.path.abspath(args.models_dir)
    if kind != "auto":
        miss = missing_models(models_dir, models_needed(kind, args.tracker))
        if miss:
            log(f"Error: the {kind} pipeline needs model files that are not in "
                f"{models_dir}:")
            for m in miss:
                log("  " + m)
            log("Run `python lidar_cli.py --download-models` on a computer with "
                "internet access, then copy the models/ folder here.")
            return 3

    work = tempfile.mkdtemp(prefix="lidar_cli_")
    progress = Progress(quiet=q)
    try:
        # --- Get the session into a private work folder ---------------------
        # Always a copy: the pipelines write caches and charts/ next to the
        # session, and the original recording must not be changed.
        if os.path.isdir(src):
            log(f"Copying session folder {src} ...", q)
            dest = os.path.join(work, os.path.basename(src.rstrip("\\/")) or "session")
            shutil.copytree(src, dest, ignore=shutil.ignore_patterns(
                "*.zip", "__pycache__", "charts"))
        else:
            log(f"Extracting {src} ...", q)
            try:
                extract_zip(src, work)
            except zipfile.BadZipFile:
                log(f"Error: {args.input} is not a valid ZIP file.")
                return 2
        session = find_session(work)
        if session is None:
            log("Error: no Stray Scanner session found. Expected a folder "
                "containing rgb.mp4, camera_matrix.csv, depth/ and confidence/.")
            return 2

        auto = kind == "auto"
        if auto:
            log("Detecting subject (whole body or hand close-up) ...", q)
            kind, explain = detect_kind(work, session, models_dir)
            log(f"  -> {kind}. {explain}", q)
            miss = missing_models(models_dir, models_needed(kind, args.tracker))
            if miss:
                log("Error: missing model files: " + ", ".join(miss) +
                    ". Run `python lidar_cli.py --download-models` on a "
                    "computer with internet access.")
                return 3

        model_desc = {
            "hand": "MediaPipe Hands",
            "foot": "RTMPose-WholeBody (RTMW, rtmlib)",
            "toe_tap": "RTMPose-WholeBody (RTMW, rtmlib)",
            "spine": "SpinePose-L (spinepose)",
            "rest_tremor": "MediaPipe Pose",
            "torso": ("RTMPose-m (rtmlib)" if args.tracker == "rtmpose"
                      else "MediaPipe Pose"),
        }[kind]
        label = f"{test_id} {label_of(test_id)}" if test_id else kind
        log(f"Analysing: {label} | pipeline {kind} | {model_desc}", q)

        script = build_script(kind, args.tracker, session, test_id, models_dir,
                              videos=not args.no_videos)
        progress(0, f"Starting {kind} pose estimation ...")
        run_pipeline(script, work, models_dir, args.timeout, progress, args.verbose)
        progress(99, "Bundling results ...")
        progress.done()

        # --- Bundle ----------------------------------------------------------
        chart_dir = os.path.join(work, "charts", session)
        data_dir = os.path.join(chart_dir, "data")
        if not os.path.isdir(data_dir):
            raise RuntimeError("The pipeline ran but produced no CSV files. "
                               "Check that the video and LiDAR frames are valid.")
        csv_files = [(n, os.path.join(data_dir, n))
                     for n in sorted(os.listdir(data_dir)) if n.endswith(".csv")]
        if not csv_files:
            raise RuntimeError("The pipeline produced no CSV files.")

        videos = []
        for role, work_name, out_name in (("skeleton", MOVIE_NAME, skel_name),
                                          ("rgb", RGB_NAME, rgb_name),
                                          ("lidar", DEPTH_NAME, depth_name)):
            p = os.path.join(work, work_name)
            if os.path.isfile(p):
                videos.append((role, out_name, p))

        graphs = []
        gdir = os.path.join(chart_dir, "graphs")
        if args.graphs and os.path.isdir(gdir):
            graphs = [(n, os.path.join(gdir, n)) for n in sorted(os.listdir(gdir))
                      if n.lower().endswith(".png")]

        frames = csv_stats(csv_files[0][1])[0]
        info = {
            "kind": kind, "auto": auto, "test_id": test_id or None,
            "model_desc": model_desc,
            "input_name": os.path.basename(src.rstrip("\\/")),
            "frames": frames,
            "when": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "version": git_version(),
            "command": "python lidar_cli.py " + " ".join(sys.argv[1:]),
        }
        readme = build_readme(info, csv_files,
                              [(r, n) for r, n, _ in videos], graphs)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        tmp_out = out_path + ".partial"
        with zipfile.ZipFile(tmp_out, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("README.txt", readme.replace("\n", "\r\n"))
            for name, path in csv_files:
                zf.write(path, name)
            for _, name, path in videos:
                zf.write(path, name, compress_type=zipfile.ZIP_STORED)
            for name, path in graphs:
                zf.write(path, "graphs/" + name)
        os.replace(tmp_out, out_path)

        mins, secs = divmod(int(time.monotonic() - t0), 60)
        log(f"\nDone in {mins}m {secs}s: {out_path}", q)
        log(f"  {len(csv_files)} CSV files, {len(videos)} videos"
            + (f", {len(graphs)} graphs" if graphs else "") + ", README.txt", q)
        if args.json:
            print(json.dumps({"output": out_path, "test": test_id or None,
                              "pipeline": kind, "frames": frames,
                              "csv_files": [n for n, _ in csv_files],
                              "videos": [n for _, n, _ in videos]}))
        else:
            print(out_path)
        return 0
    except RuntimeError as exc:
        progress.done()
        log(f"Error: {exc}")
        return 1
    finally:
        if args.keep_work:
            log(f"Work folder kept: {work}", q)
        else:
            shutil.rmtree(work, ignore_errors=True)


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="lidar_cli.py",
        description="Analyse a Stray Scanner LiDAR recording for an MDS-UPDRS "
                    "Part 3 test. Works fully offline; produces a ZIP of CSVs, "
                    "videos and a README.",
        epilog="Examples:\n"
               "  python lidar_cli.py --list-tests\n"
               "  python lidar_cli.py recording.zip --test 3.10\n"
               "  python lidar_cli.py recording.zip --test 3.4a -o out/p07_3.4a.zip\n"
               "  python lidar_cli.py --download-models   (once, on a computer "
               "with internet)",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", nargs="?",
                    help="Stray Scanner recording: a .zip, or an unzipped "
                         "session folder (rgb.mp4, depth/, confidence/, "
                         "camera_matrix.csv)")
    what = ap.add_mutually_exclusive_group()
    what.add_argument("-t", "--test", choices=list(TESTS), metavar="ID",
                      help="MDS-UPDRS Part 3 item, e.g. 3.10 or 3.4a "
                           "(see --list-tests). Picks the pipeline and names "
                           "the output.")
    what.add_argument("-p", "--pipeline", choices=KINDS,
                      help="Choose a pipeline directly instead of a test "
                           "(default when no --test: auto, which tells a "
                           "walking person from a hand close-up)")
    ap.add_argument("--tracker", choices=TRACKERS, default="mediapipe",
                    help="Pose tracker for the gait pipeline (tests 3.9-3.12). "
                         "Default: mediapipe")
    ap.add_argument("-o", "--output",
                    help="Output .zip path or folder. Default: named after the "
                         "test, e.g. 3.10_Gait.zip, in the current folder")
    ap.add_argument("--overwrite", action="store_true",
                    help="Replace the output file if it already exists")
    ap.add_argument("--no-videos", action="store_true",
                    help="Skip the skeleton/camera/LiDAR videos (much faster; "
                         "CSVs are unchanged)")
    ap.add_argument("--graphs", action="store_true",
                    help="Also include PNG plots of each angle and joint over "
                         "time (not in the website's download)")
    ap.add_argument("--models-dir", default=os.environ.get("LIDAR_MODELS_DIR",
                                                           DEFAULT_MODELS_DIR),
                    help="Folder holding downloaded model weights. Default: "
                         "models/ next to this script")
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                    help=f"Give up after this many seconds. Default: {DEFAULT_TIMEOUT}")
    ap.add_argument("--list-tests", action="store_true",
                    help="List the MDS-UPDRS tests and exit")
    ap.add_argument("--download-models", action="store_true",
                    help="Download the RTMPose/SpinePose weights into the "
                         "models folder (needs internet; run once) and exit")
    ap.add_argument("--json", action="store_true",
                    help="Print a JSON summary instead of just the output path")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="Show the pipeline's full output")
    ap.add_argument("-q", "--quiet", action="store_true",
                    help="Print only the output path (or errors)")
    ap.add_argument("--keep-work", action="store_true",
                    help="Keep the temporary work folder (for debugging)")
    args = ap.parse_args(argv)

    if args.list_tests:
        list_tests()
        return 0
    if args.download_models:
        return download_models(os.path.abspath(args.models_dir))
    if not args.input:
        ap.error("an input recording is required (or use --list-tests / "
                 "--download-models)")
    return analyse(args)


if __name__ == "__main__":
    sys.exit(main())
