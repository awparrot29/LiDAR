"""One program for both subjects: recording in, joint coordinates and a stick
figure out.

    python process_session.py <folder or .zip>              # auto-detect
    python process_session.py session.zip --kind hand       # force hand
    python process_session.py session.zip --kind torso      # force torso
    python process_session.py session.zip --no-movie -o results

Writes, under <out>/:
    data/<landmark>.csv      x, y, z in metres, one row per frame
    data/<joint> angle.csv   degrees, one row per frame
    graphs/*.png             angle and z-distance traces
    skeleton.mp4             two-view 3D stick figure, to verify the output
    summary.txt              tracking rate and per-angle ranges

Subject kind is detected from the recording — lower-body visibility separates a
torso from a close-up hand cleanly (measured 0.99 against 0.00). `--kind` overrides
it, and the detection evidence is always printed so a wrong call is visible rather
than silent.
"""
import argparse
import os
import shutil
import tempfile
import zipfile

import angles
import detect
import extract
import profiles
import skeleton3d

MOVIE_NAME = "skeleton.mp4"


def extract_zip(path, dest):
    """Extract, tolerating Windows-style backslash entry names.

    Some Windows zip tools store entries as `depth\\000000.png`; the stdlib treats
    that as a filename, so no depth/ folder appears and the session looks invalid.
    Path components are filtered so a crafted entry cannot escape dest.
    """
    with zipfile.ZipFile(path) as zf:
        for info in zf.infolist():
            name = info.filename.replace("\\", "/")
            if name.endswith("/") or "__MACOSX" in name:
                continue
            parts = [p for p in name.split("/") if p not in ("", ".", "..")]
            if not parts:
                continue
            target = os.path.join(dest, *parts)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
    for dirpath, dirs, files in os.walk(dest):
        dirs[:] = [d for d in dirs if d != "__MACOSX"]
        if ("rgb.mp4" in files and "camera_matrix.csv" in files
                and "depth" in dirs):
            return dirpath
    raise SystemExit("No Stray Scanner recording found inside the zip")


def run(session, kind=None, out_dir=None, movie=True, graphs=True, max_hands=1):
    """Process one already-extracted session folder. Returns the output path."""
    if kind is None:
        kind, ev = detect.detect(session)
        print(detect.explain(ev))
    else:
        print(f"Subject kind forced to '{kind}' (detection skipped)")
    profile = profiles.get(kind)

    arrays, geom = extract.extract_all_landmarks(session, kind=kind,
                                                 max_hands=max_hands)
    fps = 60.0
    angle_series = angles.compute(arrays, profile)

    out_dir = out_dir or os.path.join(session, "output", kind)
    os.makedirs(out_dir, exist_ok=True)
    angles.write(arrays, angle_series, profile, out_dir, fps=fps, graphs=graphs)

    text = angles.summary(arrays, angle_series, profile)
    with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as fh:
        fh.write(f"{kind} — {os.path.basename(os.path.normpath(session))}\n")
        fh.write(text + "\n")
    print("\n" + text)

    if movie:
        try:
            path = skeleton3d.render(arrays, profile,
                                     os.path.join(out_dir, MOVIE_NAME), fps=fps)
            print(f"\nstick figure: {path}")
        except Exception as exc:
            print(f"\nstick figure skipped ({type(exc).__name__}: {exc})")
    return out_dir


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("session", help="session folder or .zip")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--kind", choices=(profiles.TORSO, profiles.HAND),
                    default=None,
                    help="skip auto-detection and force the subject kind")
    ap.add_argument("--no-movie", action="store_true")
    ap.add_argument("--no-graphs", action="store_true")
    ap.add_argument("--max-hands", type=int, default=1)
    args = ap.parse_args(argv)

    tmp = None
    session = args.session
    try:
        if os.path.isfile(session) and zipfile.is_zipfile(session):
            tmp = tempfile.mkdtemp(prefix="motion_")
            print(f"Extracting {os.path.basename(session)} ...")
            session = extract_zip(session, tmp)
        if not os.path.isdir(session):
            raise SystemExit(f"Not a session folder or zip: {args.session}")

        out = args.out
        if out is None and tmp is not None:
            # a zip has nowhere sensible to write inside itself
            base = os.path.splitext(os.path.basename(args.session))[0]
            out = os.path.join(os.path.dirname(os.path.abspath(args.session)),
                               f"{base}_results")
        out_dir = run(session, kind=args.kind, out_dir=out,
                      movie=not args.no_movie, graphs=not args.no_graphs,
                      max_hands=args.max_hands)
        print(f"\nOutput: {out_dir}")
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
