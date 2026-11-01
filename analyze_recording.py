#!/usr/bin/env python3
"""Analyse a LiDAR recording of an MDS-UPDRS Part 3 test - offline.

Does what the LiDar website does, with no web server and no internet
connection. Give it a Stray Scanner recording and the test it was recorded
for; it writes a ZIP of joint-position and joint-angle spreadsheets (CSV),
three check videos, and a README that explains every file.

    python analyze_recording.py RECORDING --test TEST_ID

    python analyze_recording.py --list-tests
    python analyze_recording.py recording.zip --test 3.10
    python analyze_recording.py recording.zip --test 3.4a -o results/p07_3.4a.zip

Full setup and usage instructions: README.md, section "Analysing recordings
offline".

Every option on the website is an argument here:

    website                       command line
    -------                       ------------
    choose recording (.zip)       RECORDING  (a .zip, or an unzipped folder)
    MDS-UPDRS Part 3 Test menu    --test 3.10
    Tracker (MediaPipe/RTMPose)   --tracker mediapipe | rtmpose
    Download results              --output results.zip

Offline models: several models are normally downloaded on first use, which
fails behind a firewall: MediaPipe's high-accuracy pose model (gait tests
3.9-3.12 and rest tremor 3.17), the RTMPose models (3.7, 3.8, --tracker
rtmpose) and the SpinePose model (3.13). Only the hand tests use nothing
extra. Run
`python analyze_recording.py --download-models` once on a computer with
internet access and copy the resulting models/ folder across; this program
loads models only from there and never touches the network.
"""
import argparse
import collections
import csv
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
import zipfile
from datetime import datetime

PROG = "analyze_recording.py"
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

# Items a clinician may type that this program cannot measure.
NOT_MEASURABLE = {
    "3.1": "Speech", "3.2": "Facial expression", "3.3": "Rigidity",
    "3.14": "Global spontaneity of movement",
    "3.17e": "Rest tremor amplitude - lip/jaw",
    "3.18": "Constancy of rest tremor",
}

KINDS = ("torso", "hand", "foot", "toe_tap", "rest_tremor", "spine", "auto")
MOTION_KINDS = ("hand", "foot", "toe_tap", "rest_tremor", "spine")
TRACKERS = ("mediapipe", "rtmpose")

KIND_INFO = {
    "torso": "Whole body: 12 joints (shoulders, elbows, wrists, hips, knees, "
             "ankles) and 6 joint angles.",
    "hand": "One hand: 21 points (the wrist plus 4 on each finger and thumb) "
            "and 15 finger joint angles.",
    "foot": "Whole body plus feet: 18 points (the 12 body joints plus heel, "
            "big toe and small toe on each foot) and 8 angles, including "
            "ankle dorsiflexion.",
    "toe_tap": "Lower legs only: 10 points (knees, ankles, heels, big and "
               "small toes) and both ankle dorsiflexion angles.",
    "rest_tremor": "Whole body: the same 12 joints and 6 angles as the gait "
                   "tests, smoothed less so that 4-6 Hz tremor is kept.",
    "spine": "Spine and posture: 14 points (5 levels along the spine, neck, "
             "head, hips, shoulders, knees) and 6 spinal section angles.",
}

# Plain-language explanation of the point names, per pipeline.
GLOSSARY = {
    "hand": [
        "Each finger has four points, from the hand outward: MCP (the "
        "knuckle, where the finger meets the hand), PIP (the middle finger "
        "joint), DIP (the joint nearest the nail) and tip (the end of the "
        "finger).",
        "The thumb has CMC (the base of the thumb, near the wrist), MCP (the "
        "thumb's knuckle), IP (its one finger joint) and tip.",
        "Only ONE hand is tracked. The file names do not say which hand; the "
        "test does (a = right hand, b = left hand). Keep the other hand out "
        "of view when recording.",
    ],
    "spine": [
        "spine_01 to spine_05 are five points along the back, from the lower "
        "back (spine_01, lumbar) up to between the shoulder blades "
        "(spine_05, upper thoracic). They are estimated from the shape of "
        "the body in the image, not located vertebrae, so treat them as "
        "approximate levels.",
        "hip is the midpoint between the two hips; neck is the base of the "
        "neck; head is a point on the head.",
    ],
    "foot": [
        "big_toe and small_toe are the tips of those toes; heel is the back "
        "of the heel. The other names are the usual joints.",
    ],
    "toe_tap": [
        "big_toe and small_toe are the tips of those toes; heel is the back "
        "of the heel.",
    ],
    "torso": ["The names are the usual joints (shoulder, elbow, wrist, hip, "
              "knee, ankle)."],
    "rest_tremor": ["The names are the usual joints (shoulder, elbow, wrist, "
                    "hip, knee, ankle)."],
}

# Which files matter for each test, and a worked example of using them.
TEST_GUIDE = {
    "hand_tapping": (
        "Finger tapping: the distance between thumb_tip and index_tip, frame "
        "by frame, is the opening of each tap. Its peaks give tap size and "
        "speed, and whether taps get smaller or slower over time "
        "(decrement), hesitations and halts.",
        "Tap opening (metres) at each frame = distance between thumb_tip and "
        "index_tip (formula below)."),
    "hand_movements": (
        "Hand opening and closing: the MCP and PIP angles of the four "
        "fingers rise as the hand opens and fall as it closes. The size of "
        "each swing is the amplitude; the number of swings per second is "
        "the speed.",
        "Hand opening (metres) at each frame = distance between wrist and "
        "middle_tip."),
    "pronation": (
        "Pronation-supination: as the hand turns, the thumb_tip and "
        "pinky_tip swap sides relative to the wrist. Their x_m (left-right) "
        "positions relative to the wrist cross over with each turn.",
        "Rotation signal at each frame = thumb_tip x_m minus pinky_tip x_m. "
        "It changes sign with each turn."),
    "postural_tremor": (
        "Tremor: the fingertip and wrist positions are smoothed only "
        "lightly (3 frames, 50 ms) so tremor at 4-8 Hz is kept. The size of "
        "the small, fast back-and-forth movement of index_tip and wrist is "
        "the tremor amplitude; a frequency analysis gives the tremor "
        "frequency.",
        "Tremor amplitude: take index_tip, subtract a slow moving average "
        "(e.g. over 30 rows = 0.5 s) from each of x_m, y_m, z_m, and measure "
        "the size of what remains."),
    "toe_tap": (
        "Toe tapping: the big_toe and heel positions and the ankle "
        "dorsiflexion angle show each tap. About 90 degrees is a neutral "
        "ankle; the angle gets smaller as the toes lift toward the shin. "
        "Use the side being tested (a = right, b = left).",
        "Toe lift (metres) at each frame = the big_toe's y_m subtracted from "
        "the heel's y_m (positive = toes higher than heel)."),
    "leg_agility": (
        "Leg agility: the knee and ankle positions show how high each stomp "
        "lifts; the hip and knee angles show the range of motion. Use the "
        "side being tested (a = right, b = left).",
        "Lift height at each frame = how far the knee's y_m is above its "
        "value when the foot is on the floor (remember: smaller y_m = "
        "higher)."),
    "gait": (
        "Gait and chair rise: knee and hip angles show the range of motion "
        "of each leg; ankle positions show the steps; wrist positions show "
        "arm swing. Recordings are of walks toward or away from the camera, "
        "so walking distance shows up mainly in z_m.",
        "Walking speed = how far the hips' z_m changes, divided by the time "
        "taken (use a stretch of steady walking). Steps = the right ankle's "
        "z_m subtracted from the left ankle's: it changes sign each time the "
        "other foot moves in front, so the time between sign changes is the "
        "step time. Arm swing = how far each wrist's z_m moves back and forth "
        "relative to the same side's shoulder."),
    "posture": (
        "Posture: each spinal angle is 180 degrees when that section of the "
        "back is straight; a smaller angle means that section is more bent. "
        "Watch the skeleton video's side view to see which way (usually a "
        "forward stoop).",
        "Overall stoop: compare the thoracic angles (upper, mid, lower) "
        "with 180 degrees."),
    "rest_tremor": (
        "Rest tremor (3.17) is one recording covering all four limbs. Use "
        "the wrist files for the right and left upper extremity (3.17a "
        "RUE, 3.17b LUE) and the ankle files for the right and left lower "
        "extremity (3.17c RLE, 3.17d LLE). Positions are smoothed only "
        "lightly (3 frames, 50 ms) so tremor at 4-6 Hz is kept. Lip/jaw "
        "tremor (3.17e) is not measured.",
        "Tremor amplitude: take a wrist file, subtract a slow moving average "
        "(e.g. over 30 rows = 0.5 s) from each of x_m, y_m, z_m, and measure "
        "the size of what remains."),
}
TEST_GUIDE_KEY = {
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
    # MediaPipe Pose at model_complexity=2 (gait, rest tremor). Not in the
    # pip wheel: MediaPipe downloads it into its own package folder on first
    # use. Found missing by installing into a fresh offline environment.
    "mediapipe_heavy": ("mediapipe", ["pose_landmark_heavy.tflite"]),
}


def models_needed(kind, tracker):
    """Keys of MODEL_FILES a run needs. MediaPipe Hands (hand tests) and the
    auto-detect step use only models bundled in the MediaPipe package."""
    if kind == "rest_tremor" or (kind == "torso" and tracker == "mediapipe"):
        return ["mediapipe_heavy"]
    if kind in ("foot", "toe_tap"):
        return ["rtmpose_wholebody"]
    if kind == "spine":
        return ["spinepose"]
    if kind == "torso" and tracker == "rtmpose":
        return ["rtmpose_body"]
    return []


def checkpoint_dir(models_dir, lib):
    """Where a library's weights live under models/ (mirrors its own cache)."""
    if lib == "mediapipe":
        return os.path.join(models_dir, "mediapipe")
    return os.path.join(models_dir, lib, "hub", "checkpoints")


def mediapipe_model_dir():
    """Folder inside the installed MediaPipe package where it looks for (and
    would download) the heavy pose model. Found without importing it."""
    import importlib.util
    spec = importlib.util.find_spec("mediapipe")
    if spec is None or not spec.submodule_search_locations:
        return None
    return os.path.join(list(spec.submodule_search_locations)[0],
                        "modules", "pose_landmark")


def missing_models(models_dir, keys):
    out = []
    for key in keys:
        lib, files = MODEL_FILES[key]
        for f in files:
            if os.path.isfile(os.path.join(checkpoint_dir(models_dir, lib), f)):
                continue
            if lib == "mediapipe" and os.path.isfile(
                    os.path.join(mediapipe_model_dir() or "", f)):
                continue            # already installed in the package
            out.append(os.path.relpath(
                os.path.join(checkpoint_dir(models_dir, lib), f), models_dir))
    return out


def install_mediapipe_model(models_dir):
    """Put the heavy pose model where MediaPipe looks for it, so MediaPipe
    never tries to download it. A no-op when it is already there."""
    dest_dir = mediapipe_model_dir()
    name = MODEL_FILES["mediapipe_heavy"][1][0]
    if dest_dir is None or os.path.isfile(os.path.join(dest_dir, name)):
        return
    try:
        shutil.copy2(os.path.join(checkpoint_dir(models_dir, "mediapipe"), name),
                     os.path.join(dest_dir, name))
    except OSError as exc:
        raise UserError(f"Could not install {name} into MediaPipe's folder "
                        f"({dest_dir}): {exc}")


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
class UserError(Exception):
    """A problem the person running the program can fix; shown without a
    traceback."""


def safe(name):
    """Filename-safe text. Same rules as motion-analysis/csvout.safe()."""
    s = str(name).strip()
    s = re.sub(r"[‒-―]", " ", s)      # figure/en/em dashes
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^\w.\-]", "", s)
    s = re.sub(r"_{2,}", "_", s).strip("_")
    return s or "unnamed"


def label_of(test_id):
    """Test label with a plain hyphen, for consoles and README text."""
    return TESTS[test_id][0].replace("–", "-")


def output_base(test_id):
    """`3.10_Gait` - the stem the website uses for its downloads."""
    return safe(f"{test_id} {TESTS[test_id][0]}") if test_id else "results"


def resolve_test(text):
    """Turn what a person typed after --test into a test ID, or explain."""
    raw = text.strip()
    t = raw.lower().replace(" ", "")
    t = re.sub(r"^(mds-?updrs|item|part)?:?", "", t)
    if t in TESTS:
        return t
    if t in NOT_MEASURABLE or re.fullmatch(r"3\.3[a-e]", t):
        name = NOT_MEASURABLE.get(t, "Rigidity")
        raise argparse.ArgumentTypeError(
            f"MDS-UPDRS {t} ({name}) cannot be measured from a skeleton, so "
            "this program does not analyse it.")
    if re.fullmatch(r"3\.17[a-d]", t):
        raise argparse.ArgumentTypeError(
            f"Rest tremor is one recording of all four limbs: use --test 3.17 "
            f"(not {raw}). The results have separate files for each limb.")
    if t + "a" in TESTS:
        raise argparse.ArgumentTypeError(
            f"Test {t} has a right and a left version: use --test {t}a for "
            f"the right side ({label_of(t + 'a')}) or --test {t}b for the left "
            f"side ({label_of(t + 'b')}).")
    words = re.findall(r"[a-z]+", raw.lower())
    if words:
        hits = [tid for tid in TESTS
                if all(w in label_of(tid).lower() for w in words)]
        if hits:
            options = ", ".join(f"--test {h} ({label_of(h)})" for h in hits[:6])
            raise argparse.ArgumentTypeError(
                f'"{raw}" is not a test number. Did you mean: {options}?')
    raise argparse.ArgumentTypeError(
        f'"{raw}" is not a test this program knows. Use the item number, '
        f"for example --test 3.10 or --test 3.4a. To see them all, run: "
        f"python {PROG} --list-tests")


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


# Pipeline output lines worth keeping for the README's data-quality section.
_QUALITY_RE = re.compile(
    r"^(Warning:|ERROR:|Usable depth:|Rejected |Session recorded in|"
    r"Subject: |Background model built|Loaded from cache|Ignoring stale)")


def run_pipeline(script, work, models_dir, timeout, progress, verbose):
    """Run the pipeline child, turning its @@ markers into progress, exactly
    as the website's progress bar does. Returns the data-quality messages."""
    proc = subprocess.Popen([sys.executable, "-u", "-c", script], cwd=work,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace",
                            bufsize=1, env=child_env(models_dir))
    tail = collections.deque(maxlen=100)
    quality = []
    n_missing_depth = 0
    deadline = time.monotonic() + timeout
    TRACK, ANGLE, RENDER, MID = 55, 60, 90, 94
    tracked = False
    for raw in proc.stdout:
        tail.append(raw)
        line = raw.strip()
        if verbose:
            progress.done()
            print("  | " + line, file=sys.stderr, flush=True)
        if _QUALITY_RE.match(line) and line not in quality:
            quality.append(line)
        elif line.startswith("Missing depth at frame"):
            n_missing_depth += 1
        try:
            if line.startswith("@@FRAME "):
                done, total = (int(v) for v in line[8:].split("/"))
                if total:
                    tracked = True
                    progress(TRACK * done / total,
                             f"Tracking movement - frame {done}/{total}")
            elif line.startswith("@@JOINT "):
                frac, name = line[8:].split(" ", 1)
                cur, n = (int(v) for v in frac.split("/"))
                if tracked:
                    progress(TRACK + (ANGLE - TRACK) * cur / n,
                             f"Computing angles - {name} ({cur}/{n})")
                else:
                    progress(TRACK * (cur - 1) / n,
                             f"Loading saved tracking {cur}/{n} - {name}")
            elif line.startswith("@@RENDER "):
                done, total = (int(v) for v in line[9:].split("/"))
                if total:
                    progress(ANGLE + (RENDER - ANGLE) * done / total,
                             f"Making skeleton video - frame {done}/{total}")
            elif line.startswith("@@PREVIEW "):
                label, frac = line[10:].split(" ", 1)
                done, total = (int(v) for v in frac.split("/"))
                lo, hi = (RENDER, MID) if label == "rgb" else (MID, 98)
                what = "camera" if label == "rgb" else "LiDAR"
                if total:
                    progress(lo + (hi - lo) * done / total,
                             f"Making {what} video - frame {done}/{total}")
        except ValueError:
            pass
        if time.monotonic() > deadline:
            proc.kill()
            raise RuntimeError(f"Stopped after {timeout // 60} minutes. For a "
                               "very long recording, allow more time with "
                               "--timeout (in seconds).")
    if proc.wait() != 0:
        progress.done()
        raise RuntimeError("The analysis failed. Last output:\n"
                           + "".join(tail)[-3000:])
    if n_missing_depth:
        quality.append(f"Missing depth at {n_missing_depth} frame(s)")
    return quality


# ---------------------------------------------------------------------------
# README
# ---------------------------------------------------------------------------
WRAP = 78


def para(text, indent=0, bullet=None):
    """Wrapped paragraph for the README (plain text, read in Notepad)."""
    pad = " " * indent
    first = pad + (bullet + " " if bullet else "")
    rest = pad + (" " * (len(bullet) + 1) if bullet else "")
    return textwrap.fill(text, WRAP, initial_indent=first,
                         subsequent_indent=rest, break_on_hyphens=False)


def heading(text):
    return f"\n{text}\n{'-' * len(text)}"


def profile_tables(kind):
    """(landmark names, {angle name: (pivot, a, c)}, profile) for a kind."""
    sys.path.insert(0, MOTION_DIR)
    try:
        import profiles  # imports mediapipe; fine, it is installed for the run
        p = profiles.get(kind)
        return list(p["landmarks"]), dict(p["angles"]), p
    finally:
        sys.path.remove(MOTION_DIR)


def csv_stats(path):
    """(rows, header, rows with no data). A row has no data when its values
    are blank/nan, or a position is exactly 0,0,0 (the pipelines' marker for
    'never measured')."""
    with open(path, newline="", encoding="utf-8") as fh:
        rd = csv.reader(fh)
        header = next(rd, [])
        rows = empty = 0
        for row in rd:
            rows += 1
            vals = row[1:] if header[:1] == ["time_s"] else row
            nums = []
            for v in vals:
                try:
                    nums.append(float(v))
                except ValueError:
                    nums.append(float("nan"))
            if (not nums or any(x != x for x in nums)
                    or (len(nums) == 3 and all(x == 0 for x in nums))):
                empty += 1
    return rows, header, empty


def session_geometry(session_dir):
    """Camera geometry (frame size and lens calibration) of a recording, as
    the pipelines use it, or None if it cannot be read."""
    sys.path.insert(0, MOTION_DIR)
    try:
        import sessiongeom
        return sessiongeom.frame_geometry(session_dir)
    except Exception:
        return None
    finally:
        sys.path.remove(MOTION_DIR)


def edge_spans(path, geom):
    """Frames where a position sits on the edge of the picture.

    The pipelines clamp a point that is outside the image to the nearest
    edge pixel, so such a point gets a position that is not real. Projecting
    each 3D position back into the image finds those frames. Returns
    (count, [(start_s, end_s), ...]) or None when geometry is unknown.
    """
    if not geom:
        return None
    fx, fy, cx, cy = geom["fx"], geom["fy"], geom["cx"], geom["cy"]
    w, h = geom["width"], geom["height"]
    times = []
    with open(path, newline="", encoding="utf-8") as fh:
        rd = csv.reader(fh)
        next(rd, None)
        for row in rd:
            try:
                t, x, y, z = (float(v) for v in row[:4])
            except ValueError:
                continue
            if not z > 0:
                continue
            col, r = x * fx / z + cx, y * fy / z + cy
            if col <= 0.5 or col >= w - 1.5 or r <= 0.5 or r >= h - 1.5:
                times.append(t)
    spans = []
    for t in times:
        if spans and t - spans[-1][1] <= 2.5 / FPS:
            spans[-1][1] = t
        else:
            spans.append([t, t])
    return len(times), [(a, b) for a, b in spans]


def fmt_spans(spans, limit=3):
    parts = [f"{a:.1f}-{b:.1f} s" if b - a >= 0.05 else f"{a:.1f} s"
             for a, b in spans[:limit]]
    if len(spans) > limit:
        parts.append(f"and {len(spans) - limit} more")
    return ", ".join(parts)


def describe_csv(fname, test_id, landmarks, angle_defs):
    """(type, name, description) for one CSV, from its file name."""
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
            scale = "180 = straight; smaller = more bent."
        return ("angle", name,
                f"{title}: the angle at the {pivot} between the line to the "
                f"{a} and the line to the {c}. {scale}")
    if stem in by_safe_lm:
        name = by_safe_lm[stem]
        return ("point", name, f"Position of the {name}.")
    if stem.endswith("_angle"):
        name = stem[:-6].replace("_", " ")
        return ("angle", name, f"{name[0].upper() + name[1:]} angle.")
    name = stem.replace("_", " ")
    return ("point", name, f"Position of the {name}.")


def quality_summary(quality, kind):
    """Plain-language bullets from the pipeline's own quality messages."""
    who = "hand" if kind == "hand" else "person"
    out, raw = [], []
    tracked_fresh = any(q.startswith("Usable depth:") for q in quality)
    if tracked_fresh and not any(re.match(r"Warning: no \w+ detected", q)
                                 for q in quality):
        out.append(f"The {who} was found by the pose model in every frame.")
    for line in quality:
        m = re.match(r"Rejected (\d+) readings that were the background "
                     r"\(([\d.]+)% of samples; most affected: (.+), (\d+)\)", line)
        if m:
            out.append(
                f"{m.group(1)} distance readings ({m.group(2)}%) landed on the "
                "background behind the body (for example the wall or floor, "
                "seen past an arm or leg) and were discarded. Most affected: "
                f"{m.group(3)} ({m.group(4)} frames). These are part of the "
                "readings that were not usable above.")
            continue
        m = re.match(r"Rejected (\d+) readings on static-wall pixels "
                     r"\(([\d.]+)% of samples", line)
        if m:
            out.append(
                f"{m.group(1)} distance readings ({m.group(2)}%) landed on parts "
                "of the picture that never change (walls) and were discarded.")
            continue
        m = re.match(r"Rejected (\d+) readings more than (\d+) cm in depth "
                     r"from the rest of the subject \(([\d.]+)%", line)
        if m:
            out.append(
                f"{m.group(1)} distance readings ({m.group(3)}%) were more than "
                f"{m.group(2)} cm nearer or further than the rest of the body, "
                "which is not physically possible, and were discarded.")
            continue
        m = re.match(r"Warning: no \w+ detected on (\d+)/(\d+) frames", line)
        if m:
            n, tot = int(m.group(1)), int(m.group(2))
            out.append(
                f"The {who} was NOT found in {n} of {tot} frames "
                f"({100 * n / max(tot, 1):.1f}%). In those frames every point "
                "keeps its last known position, so the data is frozen (flat) "
                "there rather than missing. Check the skeleton video for "
                "moments where it stops moving.")
            continue
        m = re.match(r"Usable depth: (\d+)/(\d+) samples \(([\d.]+)%\)", line)
        if m:
            pct = float(m.group(3))
            out.append(
                f"{pct:.1f}% of the LiDAR distance readings at the tracked "
                "points were usable. The rest were missing, marked "
                "low-confidence by the sensor, or turned out to be the wall "
                "or floor behind the body; those were filled in from the "
                "nearest usable readings in time. The lower this figure, the "
                "more of the distances (z_m) are filled in rather than "
                "measured. For comparison, good recordings in this project "
                "had about 80%." + (" This recording is LOW - treat the "
                                    "results with caution, and consider "
                                    "re-recording." if pct < 20 else ""))
            continue
        m = re.match(r"Warning: camera moved (\d+) cm", line)
        if m:
            out.append(
                f"The camera moved about {m.group(1)} cm during the recording. "
                "Positions are measured from the camera, so camera movement "
                "shows up as movement of the body. Use a tripod.")
            continue
        m = re.match(r"Warning: (\d+)/(\d+) frames have no confidence map", line)
        if m:
            out.append(
                f"{m.group(1)} of {m.group(2)} frames had no LiDAR confidence "
                "information, so unreliable distance readings in those frames "
                "could not be screened out.")
            continue
        if line.startswith("Loaded from cache"):
            out.append("This recording contained saved tracking results from "
                       "an earlier analysis, and they were reused. So how "
                       "often the person was found, and how many distance "
                       "readings were usable, were not measured this time.")
            continue
        if line.startswith(("Ignoring stale", "Background model built",
                            "Session recorded in", "Subject:")):
            continue
        raw.append(line)
    return out, raw


def build_readme(info, csv_files, videos, graphs, quality):
    """README text for the output ZIP. `csv_files` is [(name, path)]."""
    kind, test_id = info["kind"], info["test_id"]
    try:
        landmarks, angle_defs, prof = profile_tables(kind)
    except Exception:
        landmarks, angle_defs, prof = [], {}, {}

    rows_info = []
    edge_points = []          # (name, frames at edge, spans)
    for fname, path in csv_files:
        typ, name, desc = describe_csv(fname, test_id, landmarks, angle_defs)
        rows, header, empty = csv_stats(path)
        edge = None
        if typ == "point":
            edge = edge_spans(path, info.get("geometry"))
            if edge and edge[0]:
                edge_points.append((name, edge[0], edge[1]))
        rows_info.append((typ, fname, desc, rows, empty, edge))
    n_angles = sum(1 for r in rows_info if r[0] == "angle")
    n_points = len(rows_info) - n_angles
    frames = info["frames"]

    L = []
    w = L.append
    title = "LiDAR Movement Analysis - Results"
    w(title)
    w("=" * len(title))
    w("")
    if test_id:
        w(f"Test:          MDS-UPDRS Part 3, item {test_id} - {label_of(test_id)}")
    else:
        w("Test:          (not given)")
    w(f"Recording:     {info['input_name']}")
    w(f"Length:        {frames} frames = {frames / FPS:.1f} seconds "
      f"(60 frames per second)")
    w(f"What was measured: {KIND_INFO.get(kind, kind)}")
    w(f"Pose model:    {info['model_desc']}")
    w(f"Processed:     {info['when']} on {platform.node() or 'this computer'}")
    w(f"Program:       {PROG} (code version {info['version']}, "
      f"Python {platform.python_version()})")
    w(f"Command:       {info['command']}")
    w("")
    w(para("These are measurements to support the clinician's rating, not a "
           "score. This is a research tool and has not been validated as a "
           "clinical measuring instrument."))

    # ---- At a glance -------------------------------------------------------
    w(heading("WHAT IS IN THIS ZIP"))
    w(para(f"{n_points} position files: where each tracked point on the body "
           "was, in 3D, in every frame.", 2, "*"))
    w(para(f"{n_angles} angle files: the angle of each joint in every frame.",
           2, "*"))
    if videos:
        w(para(f"{len(videos)} videos, to check by eye that the tracking was "
               "right.", 2, "*"))
    if graphs:
        w(para(f"{len(graphs)} graphs (in the graphs folder): quick-look "
               "plots of the same data.", 2, "*"))
    w(para("This README.", 2, "*"))
    w("")
    w(para("Every file name starts with the test number, so results from "
           "several tests can be kept in one folder without mixing them up."))

    # ---- Quick start --------------------------------------------------------
    w(heading("QUICK START"))
    w(para("1. Watch the skeleton video first. If the stick figure does not "
           "follow the person in the camera video, the numbers are not "
           "reliable.", 2))
    w(para("2. Read DATA QUALITY below.", 2))
    w(para("3. Open the files listed under WHICH FILES MATTER FOR THIS TEST "
           "in Excel (double-click), R, Python, SPSS or MATLAB.", 2))

    # ---- For this test -----------------------------------------------------
    if test_id and test_id in TEST_GUIDE_KEY:
        what, example = TEST_GUIDE[TEST_GUIDE_KEY[test_id]]
        w(heading("WHICH FILES MATTER FOR THIS TEST"))
        w(para(what))
        w("")
        w(para("Example: " + example))

    # ---- Data quality ------------------------------------------------------
    w(heading("DATA QUALITY"))
    summary, raw = quality_summary(quality, kind)
    if edge_points:
        edge_points.sort(key=lambda e: -e[1])
        parts = [f"{n} ({fmt_spans(sp, 2)})" for n, _, sp in edge_points[:6]]
        more = (f", and {len(edge_points) - 6} more"
                if len(edge_points) > 6 else "")
        summary.append(
            "Some points went outside the picture: " + "; ".join(parts) + more
            + ". In those frames the position is pinned to the edge of the "
            "picture, so it is NOT the real position - leave those times out. "
            "Angles that use those points are unreliable at the same times. "
            "Exact counts are under ALL FILES.")
    elif info.get("geometry"):
        summary.append("Every point stayed inside the picture for the whole "
                       "recording.")
    if summary:
        w("From the analysis of this recording:")
        w("")
        for s in summary:
            w(para(s, 2, "*"))
        if raw:
            w("")
            w("Other messages from the analysis:")
            for r in raw:
                w(para(r, 4))
    else:
        w(para("No quality figures were reported for this run."))
    w("")
    w(para("How gaps are handled: the files have a value in almost every "
           "row, even when the measurement was not possible in that frame. "
           "A point the pose model could not see keeps its last known "
           "position, and a distance the LiDAR could not measure is filled "
           "in from the nearest good readings. So a full column does NOT "
           "mean every value was measured. Use the figures above and the "
           "skeleton video to judge the data."))
    w("")
    w(para("Rows with no data at all are written as 0, 0, 0 (positions) or "
           "nan (angles). The 'No data' count for each file is listed under "
           "ALL FILES below. It is usually 0."))

    # ---- How positions work --------------------------------------------------
    w(heading("POSITION FILES: time_s, x_m, y_m, z_m"))
    w(para("One row per video frame. Each row says where that point was, in "
           "metres, measured from the camera:"))
    w("")
    w("    time_s  seconds since the start of the recording (row number / 60)")
    w("    x_m     distance to the RIGHT of the centre of the camera's view")
    w("            (negative = to the left)")
    w("    y_m     distance BELOW the centre of the camera's view")
    w("            (negative = above). Note: larger y_m = LOWER.")
    w("    z_m     distance AWAY from the camera (the LiDAR reading)")
    w("")
    w("        Seen from behind the camera:")
    w("")
    w("              - y (up)")
    w("                 |")
    w("      - x  ------+------  + x (right)        + z = straight ahead,")
    w("                 |                              away from the camera")
    w("              + y (down)")
    w("")
    w(para("'Left' and 'right' in x are as seen in the video (from the "
           "camera), not the person's left and right. The point names "
           "(left_knee, right_wrist...) ARE the person's own left and right."))
    w("")
    w(para("The camera is at 0, 0, 0 and must stay still (on a tripod) "
           "during the recording, so positions can be compared over time."))
    w("")
    w(para("Units are metres: 0.05 = 5 cm. The LiDAR distance is accurate to "
           "about 1-2%."))

    # ---- Angles --------------------------------------------------------------
    w(heading("ANGLE FILES: time_s, angle_deg"))
    w(para("One row per video frame. angle_deg is the angle in degrees at a "
           "joint, between the two body segments that meet there, worked "
           "out in 3D from the position files. Each file's description "
           "below says exactly which three points it uses."))
    w("")
    w(para("180 degrees means the two segments are in a straight line (a "
           "fully straight knee or elbow). Smaller means more bent. The "
           "angle has no direction: it says how much a joint is bent, not "
           "which way. The ankle dorsiflexion angle is about 90 degrees "
           "for a neutral ankle."))

    # ---- Names ---------------------------------------------------------------
    w(heading("POINT NAMES"))
    for g in GLOSSARY.get(kind, []):
        w(para(g, 2, "*"))

    # ---- Videos ----------------------------------------------------------------
    if videos:
        w(heading("VIDEOS"))
        vdesc = {
            "skeleton": "Stick figure drawn from the position files. "
                        + ("Left panel: a three-quarter view (from in front, "
                           "a little to the side). " if kind == "torso" else
                           "Left panel: the front view, as the camera saw "
                           "it. ")
                        + "Right panel: a side view, turned 90 degrees, which "
                        "shows the depth (z_m). If the stick figure does not "
                        "match the person, the numbers are not reliable.",
            "rgb": "The original camera video. PRIVACY: it shows the "
                   "participant, including their face. Handle it as "
                   "identifiable patient data.",
            "lidar": "The LiDAR distance images as a video. Near is blue, far "
                     "is red, black means the sensor got no reading there. "
                     "Black patches on the body mean missing distances for "
                     "those frames.",
        }
        for role, fname in videos:
            w(fname)
            w(para(vdesc[role], 4))
            w("")

    if graphs:
        w(heading("GRAPHS (graphs folder)"))
        w(para("*_angle.png: each angle over time. *_distance.png: a point's "
               "distance from the camera (z_m) over time. They are drawn "
               "from the same numbers as the CSV files."))

    # ---- Calculations ---------------------------------------------------------
    w(heading("USEFUL CALCULATIONS"))
    w(para("Distance between two points in the same frame (for example "
           "thumb_tip and index_tip), in metres:"))
    w("    sqrt( (x1-x2)^2 + (y1-y2)^2 + (z1-z2)^2 )")
    w(para("In Excel: open both files, copy the x_m, y_m, z_m columns of the "
           "second file next to the first (columns F, G, H), then in row 2 "
           "enter =SQRT((B2-F2)^2+(C2-G2)^2+(D2-H2)^2) and fill down.", 4))
    w("")
    w(para("Speed of a point: the distance it moves from one row to the next "
           "divided by 1/60 second (that is, multiplied by 60) gives metres "
           "per second. Because of the gap handling described under DATA "
           "QUALITY, speed is 0 wherever the point was frozen."))
    w("")
    w(para("How often something repeats (taps per second, tremor frequency): "
           "count the peaks and divide by the time in seconds, or use a "
           "frequency analysis (FFT). Rows are exactly 1/60 s apart."))

    # ---- Limits ---------------------------------------------------------------
    w(heading("THINGS TO KNOW"))
    for t in (
        "The LiDAR measures the distance to the SURFACE of the body or "
        "clothing facing the camera, so a point's z_m is a few centimetres "
        "in front of the true joint centre. Loose clothing adds to this.",
        "Points are found by a pose model in the video image. A point "
        "hidden behind the body (for example the far arm) is the model's "
        "best guess.",
        "Left and right labels can occasionally swap, for example when the "
        "person turns around. Check the skeleton video.",
        "If a point goes outside the picture (for example the feet when "
        "someone walks close to the camera), the pose model still gives a "
        "position, but it is pinned to the edge of the picture and is not "
        "real. DATA QUALITY and ALL FILES list when this happened.",
        "The image is small (192 x 256 pixels), so very small movements "
        "(a few millimetres) are near the limit of what can be measured.",
        "The LiDAR works up to about 5 m from the camera; the closer, the "
        "more detail.",
        "How the numbers are made: the pose model finds each point in the "
        "image; the LiDAR gives its distance; readings that are unreliable "
        "or belong to the background are dropped; each distance is "
        "averaged over 9 neighbouring frames (150 ms), ignoring outliers"
        + (f"; each image position is averaged over "
           f"{prof['pixel_smooth_window']} frames "
           f"({prof['pixel_smooth_window'] * 1000 / FPS:.0f} ms)"
           if prof.get("pixel_smooth_window") else "")
        + "; then the camera's lens calibration turns image position plus "
        "distance into x, y, z.",
    ):
        w(para(t, 2, "*"))

    # ---- File list ------------------------------------------------------------
    w(heading("ALL FILES"))
    w(para(f"Every CSV file has {frames} rows plus a header row. 'No data' is "
           "the number of rows with nothing measured (see DATA QUALITY)."))
    for typ, label in (("angle", "Angle files (time_s, angle_deg)"),
                       ("point", "Position files (time_s, x_m, y_m, z_m)")):
        items = [r for r in rows_info if r[0] == typ]
        if not items:
            continue
        w("")
        w(label + ":")
        w("")
        for _, fname, desc, rows, empty, edge in items:
            extra = ""
            if edge and edge[0]:
                extra = (f" Outside the picture (not real): {edge[0]} rows, "
                         f"{fmt_spans(edge[1])}.")
            w(fname)
            w(para(f"{desc} No data: {empty} of {rows} rows.{extra}", 4))
    if videos:
        w("")
        w("Videos: " + ", ".join(n for _, n in videos))
    w("")
    w(heading("OPENING THE FILES"))
    w("Excel:   double-click the .csv file.")
    w("R:       df <- read.csv(\"<file>.csv\")")
    w("Python:  import pandas as pd; df = pd.read_csv(\"<file>.csv\")")
    w("MATLAB:  T = readtable('<file>.csv');")
    w("SPSS:    File > Import Data > CSV Data")
    w("")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
def list_tests():
    print("MDS-UPDRS Part 3 tests this program can analyse.")
    print("Use the number after --test, e.g.  --test 3.10   or   --test 3.4a")
    print("(a = right side, b = left side)\n")
    for tid in TESTS:
        print(f"  {tid:<6} {label_of(tid)}")
    print("\nCannot be measured from a skeleton, so not available: 3.1 speech, "
          "3.2 facial\nexpression, 3.3 rigidity, 3.14 global spontaneity, "
          "3.17e lip/jaw tremor,\n3.18 constancy of rest tremor.")
    print("\nAll tests except the hand tests need the models folder (see "
          "README.md).")


QUICK_HELP = f"""\
Analyse a LiDAR recording of an MDS-UPDRS Part 3 test.

  python {PROG} RECORDING --test TEST

  RECORDING  the Stray Scanner recording: a .zip file, or the unzipped folder
  TEST       the MDS-UPDRS item number, e.g. 3.10 (gait) or 3.4a (finger
             tapping, right hand)

Examples:
  python {PROG} --list-tests          (shows all test numbers)
  python {PROG} patient07_gait.zip --test 3.10
  python {PROG} "C:\\Recordings\\p07 tap R.zip" --test 3.4a

All options: python {PROG} --help
"""


def download_models(models_dir):
    """Fill models/ so analysis can run with no internet. Run this on a
    machine that has internet access, then copy models/ across."""
    print(f"Model folder: {models_dir}")
    # Where each library keeps what it has already downloaded.
    defaults = {"rtmlib": os.path.expanduser("~/.cache/rtmlib/hub/checkpoints"),
                "spinepose": os.path.expanduser("~/.cache/spinepose/hub/checkpoints"),
                "mediapipe": mediapipe_model_dir() or ""}

    def copy_known():
        for key, (lib, files) in MODEL_FILES.items():
            dst_dir = checkpoint_dir(models_dir, lib)
            os.makedirs(dst_dir, exist_ok=True)
            for f in files:
                src, dst = os.path.join(defaults[lib], f), os.path.join(dst_dir, f)
                if not os.path.isfile(dst) and os.path.isfile(src):
                    print(f"  copying {f} from {defaults[lib]}")
                    shutil.copy2(src, dst)

    # 1. Reuse anything this machine has already downloaded.
    copy_known()

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
            # MediaPipe downloads this one into its own package folder.
            "import mediapipe as mp",
            "mp.solutions.pose.Pose(model_complexity=2).close()",
        ])
        r = subprocess.run([sys.executable, "-c", src], env=child_env(models_dir))
        if r.returncode != 0:
            print("Download failed (see the error above). Is this computer "
                  "online?", file=sys.stderr)
            return 1
        copy_known()

    missing = missing_models(models_dir, MODEL_FILES)
    if missing:
        print("These model files are still missing:", file=sys.stderr)
        for m in missing:
            print("  " + m, file=sys.stderr)
        return 1
    paths = {os.path.join(checkpoint_dir(models_dir, lib), f)
             for lib, files in MODEL_FILES.values() for f in files}
    total = sum(os.path.getsize(p) for p in paths)
    print(f"All models present ({total / 1e6:.0f} MB in {models_dir}).")
    print("Copy this models folder into the program folder on the offline "
          "computer.")
    return 0


def prepare_session(src, work, q):
    """Copy or extract the recording into `work`; return the session path."""
    if os.path.isdir(src):
        log(f"Copying recording folder {src} ...", q)
        dest = os.path.join(work, os.path.basename(src.rstrip("\\/")) or "session")
        shutil.copytree(src, dest, ignore=shutil.ignore_patterns(
            "*.zip", "__pycache__", "charts"))
    else:
        log(f"Unzipping {src} ...", q)
        try:
            extract_zip(src, work)
        except zipfile.BadZipFile:
            raise UserError(f"{src} is not a valid ZIP file. It may be "
                            "incomplete - try copying it from the iPad again.")
    session = find_session(work)
    if session is None:
        raise UserError(
            "This is not a Stray Scanner recording: no folder in it holds "
            "rgb.mp4, camera_matrix.csv and a depth folder. Give the whole "
            "recording (the .zip exported from Stray Scanner, or its unzipped "
            "folder), not a single video.")
    return session


def analyse(args):
    q = args.quiet
    t0 = time.monotonic()
    src = os.path.abspath(args.recording)
    if not os.path.exists(src):
        raise UserError(
            f"Cannot find {args.recording}. Check the spelling, and put the "
            "path in quotes if it contains spaces. Tip: drag the file into "
            "this window to paste its full path.")
    if os.path.isfile(src) and not src.lower().endswith(".zip"):
        raise UserError(
            f"{os.path.basename(src)} is not a .zip file. Give the whole Stray "
            "Scanner recording (a .zip, or its unzipped folder), not a single "
            "video or image.")

    test_id = args.test or ""
    kind = TESTS[test_id][1] if test_id else args.pipeline

    if args.tracker == "rtmpose" and kind not in ("torso", "auto"):
        log(f"Note: --tracker only applies to gait tests 3.9-3.12; this test "
            "uses its own model.", q)

    base = output_base(test_id or None)
    out_path = os.path.abspath(args.output or base + ".zip")
    if os.path.isdir(out_path) or (args.output or "").endswith(("/", "\\")):
        out_path = os.path.join(out_path, base + ".zip")
    if not out_path.lower().endswith(".zip"):
        out_path += ".zip"
    if os.path.exists(out_path) and not args.overwrite:
        raise UserError(f"{out_path} already exists. Choose another name with "
                        "-o, or add --overwrite to replace it.")

    models_dir = os.path.abspath(args.models_dir)

    def check_models(k):
        miss = missing_models(models_dir, models_needed(k, args.tracker))
        if miss:
            raise UserError(
                "This test needs model files that are not in the models "
                f"folder ({models_dir}):\n  " + "\n  ".join(miss) +
                f"\nOn a computer with internet access, run:  python {PROG} "
                "--download-models\nthen copy its models folder into this "
                "program's folder. (See README.md, 'Analysing recordings "
                "offline'.)", 3)
    if kind != "auto":
        check_models(kind)
        if "mediapipe_heavy" in models_needed(kind, args.tracker):
            install_mediapipe_model(models_dir)

    work = tempfile.mkdtemp(prefix="lidar_analysis_")
    progress = Progress(quiet=q)
    try:
        session = prepare_session(src, work, q)

        auto = kind == "auto"
        if auto:
            log("Detecting whether this is a whole-body or a hand recording ...", q)
            kind, explain = detect_kind(work, session, models_dir)
            log(f"  -> {kind}. {explain}", q)
            check_models(kind)
            if "mediapipe_heavy" in models_needed(kind, args.tracker):
                install_mediapipe_model(models_dir)

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
        log(f"Analysing {label}  (model: {model_desc})", q)
        log("This usually takes 3-10 minutes; add --no-videos to make it "
            "faster.", q)

        script = build_script(kind, args.tracker, session, test_id, models_dir,
                              videos=not args.no_videos)
        progress(0, "Starting ...")
        quality = run_pipeline(script, work, models_dir, args.timeout,
                               progress, args.verbose)
        progress(99, "Packing results ...")
        progress.done()

        # --- Bundle ----------------------------------------------------------
        chart_dir = os.path.join(work, "charts", session)
        data_dir = os.path.join(chart_dir, "data")
        csv_files = []
        if os.path.isdir(data_dir):
            csv_files = [(n, os.path.join(data_dir, n))
                         for n in sorted(os.listdir(data_dir)) if n.endswith(".csv")]
        if not csv_files:
            raise RuntimeError("The analysis ran but produced no results. "
                               "Check that the video and LiDAR frames are valid.")

        video_names = {"skeleton": f"{base}_skeleton.mp4",
                       "rgb": f"{base}_rgb.mp4", "lidar": f"{base}_lidar.mp4"}
        videos = []
        for role, work_name in (("skeleton", MOVIE_NAME), ("rgb", RGB_NAME),
                                ("lidar", DEPTH_NAME)):
            p = os.path.join(work, work_name)
            if os.path.isfile(p):
                videos.append((role, video_names[role], p))

        graphs = []
        gdir = os.path.join(chart_dir, "graphs")
        if args.graphs and os.path.isdir(gdir):
            graphs = [(n, os.path.join(gdir, n)) for n in sorted(os.listdir(gdir))
                      if n.lower().endswith(".png")]

        frames = csv_stats(csv_files[0][1])[0]
        info = {
            "geometry": session_geometry(os.path.join(work, session)),
            "kind": kind, "auto": auto, "test_id": test_id or None,
            "model_desc": model_desc,
            "input_name": os.path.basename(src.rstrip("\\/")),
            "frames": frames,
            "when": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "version": git_version(),
            "command": f"python {PROG} " + " ".join(sys.argv[1:]),
        }
        readme = build_readme(info, csv_files,
                              [(r, n) for r, n, _ in videos], graphs, quality)
        readme_name = f"README_{base}.txt"

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        tmp_out = out_path + ".partial"
        with zipfile.ZipFile(tmp_out, "w", zipfile.ZIP_DEFLATED) as zf:
            # CRLF + BOM so Notepad on any Windows version shows it properly.
            zf.writestr(readme_name, "\ufeff" + readme.replace("\n", "\r\n"))
            for name, path in csv_files:
                zf.write(path, name)
            for _, name, path in videos:
                zf.write(path, name, compress_type=zipfile.ZIP_STORED)
            for name, path in graphs:
                zf.write(path, "graphs/" + name)
        os.replace(tmp_out, out_path)

        mins, secs = divmod(int(time.monotonic() - t0), 60)
        log(f"\nDone in {mins} min {secs} s. Results saved to:\n  {out_path}", q)
        log(f"Inside: {len(csv_files)} spreadsheets (CSV), {len(videos)} videos"
            + (f", {len(graphs)} graphs" if graphs else "")
            + f", and {readme_name}, which explains every file.", q)
        if args.json:
            print(json.dumps({"output": out_path, "test": test_id or None,
                              "pipeline": kind, "frames": frames,
                              "readme": readme_name,
                              "csv_files": [n for n, _ in csv_files],
                              "videos": [n for _, n, _ in videos]}))
        elif q:
            print(out_path)
        return 0
    finally:
        progress.done()
        if args.keep_work:
            log(f"Work folder kept: {work}", q)
        else:
            shutil.rmtree(work, ignore_errors=True)


class _Parser(argparse.ArgumentParser):
    """Short, plain error messages instead of argparse's usage dump."""

    def error(self, message):
        message = re.sub(r"^argument -t/--test: ", "", message)
        sys.stderr.write(f"Error: {message}\n\nFor help: python {PROG} --help\n")
        sys.exit(2)


def make_parser():
    ap = _Parser(
        prog=PROG, add_help=False,
        usage=f"python {PROG} RECORDING --test TEST [options]",
        description=textwrap.fill(
            "Analyse a LiDAR recording of an MDS-UPDRS Part 3 test. Works "
            "with no internet connection. Produces a ZIP of spreadsheets "
            "(CSV), check videos, and a README that explains every file.", 78),
        epilog="Examples:\n"
               f"  python {PROG} --list-tests\n"
               f"  python {PROG} patient07_gait.zip --test 3.10\n"
               f"  python {PROG} \"C:\\Recordings\\p07 tap R.zip\" --test 3.4a "
               "-o \"C:\\Results\"\n"
               f"  python {PROG} recording.zip --test 3.17 --no-videos\n\n"
               "Put a path in quotes if it contains spaces. Full instructions: "
               "README.md",
        formatter_class=argparse.RawDescriptionHelpFormatter)

    need = ap.add_argument_group("Required")
    need.add_argument("recording", nargs="?", metavar="RECORDING",
                      help="The Stray Scanner recording: the .zip file "
                           "exported from the iPad/iPhone, or its unzipped "
                           "folder.")
    need.add_argument("-t", "--test", type=resolve_test, metavar="TEST",
                      help="The MDS-UPDRS Part 3 item this recording is for, "
                           "e.g. 3.10 (gait) or 3.4a (finger tapping, right "
                           "hand). a = right side, b = left side. "
                           "See --list-tests.")

    opt = ap.add_argument_group("Optional")
    opt.add_argument("-o", "--output", metavar="PATH",
                     help="Where to save the results: a .zip file name, or a "
                          "folder. Default: named after the test (e.g. "
                          "3.10_Gait.zip) in the current folder.")
    opt.add_argument("--no-videos", action="store_true",
                     help="Skip the three videos. Much faster; the "
                          "spreadsheets are exactly the same. Also leaves out "
                          "the camera video, which shows the participant's "
                          "face.")
    opt.add_argument("--graphs", action="store_true",
                     help="Also include quick-look graphs (PNG) of every "
                          "angle and point over time.")
    opt.add_argument("--overwrite", action="store_true",
                     help="Replace the results file if it already exists.")
    opt.add_argument("--tracker", choices=TRACKERS, default="mediapipe",
                     help="Pose model for the gait tests 3.9-3.12, as on the "
                          "website. Leave at the default (mediapipe) unless "
                          "told otherwise.")

    setup = ap.add_argument_group("Information and setup")
    setup.add_argument("-h", "--help", action="help",
                       help="Show this help and exit.")
    setup.add_argument("--list-tests", action="store_true",
                       help="List the test numbers you can use and exit.")
    setup.add_argument("--download-models", action="store_true",
                       help="One-time setup, on a computer WITH internet: "
                            "download the models that every test except the "
                            "hand tests needs into the models folder, then "
                            "exit.")

    adv = ap.add_argument_group("Advanced (normally not needed)")
    adv.add_argument("--pipeline", choices=KINDS, metavar="KIND",
                     help="Analyse without naming a test, choosing the "
                          f"analysis type directly: {', '.join(KINDS)}.")
    adv.add_argument("--models-dir", metavar="DIR",
                     default=os.environ.get("LIDAR_MODELS_DIR", DEFAULT_MODELS_DIR),
                     help="Models folder, if not 'models' next to this program "
                          "(or set LIDAR_MODELS_DIR).")
    adv.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                     metavar="SECONDS",
                     help=f"Give up after this long. Default {DEFAULT_TIMEOUT}.")
    adv.add_argument("--json", action="store_true",
                     help="Print a JSON summary when done (for scripts).")
    adv.add_argument("-v", "--verbose", action="store_true",
                     help="Show all the analysis output.")
    adv.add_argument("-q", "--quiet", action="store_true",
                     help="Print only the results path, or an error.")
    adv.add_argument("--keep-work", action="store_true",
                     help="Keep the temporary work folder (for debugging).")
    return ap


def main(argv=None):
    ap = make_parser()
    args = ap.parse_args(argv)

    if args.list_tests:
        list_tests()
        return 0
    if args.download_models:
        return download_models(os.path.abspath(args.models_dir))
    if not args.recording:
        sys.stderr.write(QUICK_HELP)
        return 2
    if args.test and args.pipeline:
        ap.error("use --test or --pipeline, not both.")
    if not args.test and not args.pipeline:
        ap.error("which test is this recording for? Add --test and the "
                 "MDS-UPDRS item number, for example --test 3.10 (gait). "
                 f"To see all numbers: python {PROG} --list-tests")
    try:
        return analyse(args)
    except UserError as exc:
        sys.stderr.write(f"\nError: {exc.args[0]}\n")
        return exc.args[1] if len(exc.args) > 1 else 2
    except RuntimeError as exc:
        sys.stderr.write(f"\nError: {exc}\n")
        return 1
    except KeyboardInterrupt:
        sys.stderr.write("\nStopped.\n")
        return 130


if __name__ == "__main__":
    sys.exit(main())
