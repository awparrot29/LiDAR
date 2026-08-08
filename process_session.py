"""Process a Stray Scanner session from the command line.

Does what the web app does, without the web app: point it at a zipped Stray
Scanner recording and it writes the joint coordinate CSVs and the 3D stick
figure movie.

    python process_session.py session.zip
    python process_session.py session.zip -o results --tracker rtmpose
    python process_session.py session.zip --keep-graphs

The coordinates are denoised the same way the site denoises them - readings
that are actually the background are rejected, then each joint's depth is
averaged over nine frames with outliers beyond two standard deviations dropped.
Device orientation is detected from the recording, so portrait and landscape
sessions both come out upright.
"""
import argparse
import collections
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
GAIT_DIR = os.path.join(HERE, "gait-analysis")
MOVIE_NAME = "gait_skeleton_3d.mp4"
TRACKERS = ("mediapipe", "rtmpose")


def extract_all(zf, dest_root):
    """Extract every member, tolerating Windows-style backslash entry names.

    Some Windows zip tools store entries as `depth\\000000.png`. The stdlib
    treats that as a filename rather than a path, so the depth/ folder never
    appears and the session looks invalid. Path components are filtered so a
    crafted entry cannot escape dest_root.
    """
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
    """Relative path to the first folder holding a Stray Scanner recording."""
    for dirpath, dirs, files in os.walk(root):
        # Skip the shadow tree a Mac/iOS "Compress" adds
        dirs[:] = [d for d in dirs if d != "__MACOSX"]
        if "rgb.mp4" in files and "camera_matrix.csv" in files and "depth" in dirs:
            return os.path.relpath(dirpath, root)
    return None


def check_tracker(tracker):
    """Fail early and clearly if the chosen tracker cannot run here."""
    if tracker != "rtmpose":
        return
    probe = subprocess.run([sys.executable, "-c", "import rtmlib"],
                           capture_output=True, text=True)
    if probe.returncode != 0:
        raise SystemExit(
            "The RTMPose tracker needs rtmlib, which failed to import:\n"
            + (probe.stderr or "").strip()
            + "\n\nInstall it with:  pip install rtmlib onnxruntime\n"
              "Or use the MediaPipe tracker:  --tracker mediapipe")


def build_script(session, tracker, want_movie):
    data_rel = os.path.join("charts", session, "data")
    lines = [
        "import sys, os",
        'import matplotlib; matplotlib.use("Agg")',
        f"sys.path.insert(0, {GAIT_DIR!r})",
    ]
    if tracker == "rtmpose":
        # calculateangle imports `pipelandmark`; swapping the module in
        # sys.modules first makes it use the RTMPose implementation instead.
        lines += ["import pipelandmark_rtmpose as _t",
                  'sys.modules["pipelandmark"] = _t']
    lines += ["import calculateangle",
              f"calculateangle.main(folder={session!r})"]
    if want_movie:
        lines += ["import skeleton3d",
                  f"_lm = skeleton3d.load_landmarks({data_rel!r})",
                  f"skeleton3d.render_movie(_lm, {MOVIE_NAME!r})"]
    return "; ".join(lines)


class Progress:
    """Renders the pipeline's @@FRAME / @@JOINT / @@RENDER markers as one bar."""

    TRACK_SHARE, ANGLE_SHARE, DONE_SHARE = 55, 60, 98

    def __init__(self, quiet):
        self.quiet = quiet
        self.pct = 0.0
        self.stage = "starting"
        self.tracked = False
        self.started = time.monotonic()

    def feed(self, line):
        try:
            if line.startswith("@@FRAME "):
                done, total = (int(v) for v in line[8:].split("/"))
                if total:
                    self.tracked = True
                    self.pct = self.TRACK_SHARE * done / total
                    self.stage = f"tracking pose {done}/{total}"
            elif line.startswith("@@JOINT "):
                frac, name = line[8:].split(" ", 1)
                k, n = (int(v) for v in frac.split("/"))
                if self.tracked:
                    self.pct = (self.TRACK_SHARE
                                + (self.ANGLE_SHARE - self.TRACK_SHARE) * k / n)
                    self.stage = f"angles {k}/{n} ({name})"
                else:
                    self.pct = self.TRACK_SHARE * (k - 1) / n
                    self.stage = f"cached joint {k}/{n} ({name})"
            elif line.startswith("@@RENDER "):
                done, total = (int(v) for v in line[9:].split("/"))
                if total:
                    self.pct = (self.ANGLE_SHARE
                                + (self.DONE_SHARE - self.ANGLE_SHARE) * done / total)
                    self.stage = f"rendering movie {done}/{total}"
            else:
                return False
        except ValueError:
            return False
        self.draw()
        return True

    def draw(self):
        if self.quiet:
            return
        width = 28
        filled = int(width * self.pct / 100)
        bar = "#" * filled + "-" * (width - filled)
        elapsed = time.monotonic() - self.started
        eta = ""
        if self.pct >= 3:
            left = elapsed * (100 - self.pct) / self.pct
            eta = f"  ~{int(left)//60}m{int(left)%60:02d}s left"
        sys.stdout.write(f"\r  [{bar}] {self.pct:5.1f}%  {self.stage:<34}{eta}   ")
        sys.stdout.flush()

    def finish(self):
        if not self.quiet:
            sys.stdout.write("\r" + " " * 100 + "\r")
            sys.stdout.flush()


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Turn a Stray Scanner ZIP into joint coordinate CSVs and a "
                    "3D stick figure movie.")
    p.add_argument("zip", help="the zipped Stray Scanner session folder")
    p.add_argument("-o", "--out", default=None,
                   help="output folder (default: <zip name>_results beside the zip)")
    p.add_argument("--tracker", choices=TRACKERS, default="mediapipe",
                   help="pose model: mediapipe (default, faster) or rtmpose "
                        "(RTMPose-m via ONNX Runtime)")
    p.add_argument("--no-movie", action="store_true",
                   help="write only the CSVs and skip the 3D render")
    p.add_argument("--keep-graphs", action="store_true",
                   help="also copy the per-joint PNG graphs the pipeline makes")
    p.add_argument("--timeout", type=int, default=3600,
                   help="give up after this many seconds (default 3600)")
    p.add_argument("-q", "--quiet", action="store_true")
    args = p.parse_args(argv)

    if not os.path.isfile(args.zip):
        raise SystemExit(f"No such file: {args.zip}")
    if not zipfile.is_zipfile(args.zip):
        raise SystemExit(f"Not a zip file: {args.zip}")
    if not os.path.isdir(GAIT_DIR):
        raise SystemExit(f"Cannot find the gait-analysis modules at {GAIT_DIR}")

    check_tracker(args.tracker)

    base = os.path.splitext(os.path.basename(args.zip))[0]
    out_dir = args.out or os.path.join(
        os.path.dirname(os.path.abspath(args.zip)), f"{base}_results")
    os.makedirs(out_dir, exist_ok=True)

    work = tempfile.mkdtemp(prefix="lidar_cli_")
    cwd = os.getcwd()
    try:
        if not args.quiet:
            print(f"Extracting {os.path.basename(args.zip)} ...")
        with zipfile.ZipFile(args.zip) as zf:
            extract_all(zf, work)

        session = find_session(work)
        if session is None:
            raise SystemExit(
                "No Stray Scanner session found in the ZIP.\n"
                "Expected a folder containing rgb.mp4, camera_matrix.csv, "
                "depth/ and confidence/.")
        if not args.quiet:
            label = session if session != "." else os.path.basename(base)
            print(f"Session: {label}   tracker: {args.tracker}")

        os.chdir(work)
        proc = subprocess.Popen(
            [sys.executable, "-u", "-c",
             build_script(session, args.tracker, not args.no_movie)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

        bar = Progress(args.quiet)
        tail = collections.deque(maxlen=100)
        deadline = time.monotonic() + args.timeout
        notes = []
        for raw in proc.stdout:
            tail.append(raw)
            line = raw.strip()
            if bar.feed(line):
                pass
            elif line.startswith(("Session recorded", "Background model",
                                  "Rejected", "Smoothing depth", "Warning:",
                                  "Ignoring stale")):
                notes.append(line)
            if time.monotonic() > deadline:
                proc.kill()
                raise SystemExit(f"Timed out after {args.timeout} s.")
        bar.finish()

        if proc.wait() != 0:
            raise SystemExit("The pipeline failed:\n" + "".join(tail)[-3000:])

        data_dir = os.path.join(work, "charts", session, "data")
        if not os.path.isdir(data_dir):
            raise SystemExit("The pipeline produced no CSV files.")

        os.chdir(cwd)
        csvs = sorted(f for f in os.listdir(data_dir) if f.endswith(".csv"))
        for name in csvs:
            shutil.copy2(os.path.join(data_dir, name), os.path.join(out_dir, name))

        movie_src = os.path.join(work, MOVIE_NAME)
        movie_out = None
        if os.path.exists(movie_src):
            movie_out = os.path.join(out_dir, f"{base}_{MOVIE_NAME}")
            shutil.copy2(movie_src, movie_out)

        graphs = 0
        if args.keep_graphs:
            gdir = os.path.join(work, "charts", session, "graphs")
            if os.path.isdir(gdir):
                target = os.path.join(out_dir, "graphs")
                os.makedirs(target, exist_ok=True)
                for name in os.listdir(gdir):
                    shutil.copy2(os.path.join(gdir, name),
                                 os.path.join(target, name))
                    graphs += 1

        if not args.quiet:
            for note in notes:
                print(f"  {note}")
            print(f"\nWrote to {out_dir}")
            print(f"  {len(csvs)} CSV files")
            if movie_out:
                mb = os.path.getsize(movie_out) / 1e6
                print(f"  {os.path.basename(movie_out)}  ({mb:.1f} MB)")
            if graphs:
                print(f"  graphs/  ({graphs} PNGs)")
    finally:
        os.chdir(cwd)
        shutil.rmtree(work, ignore_errors=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
