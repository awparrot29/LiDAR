"""Score the LiDAR pipeline against the no-LiDAR pipeline on the same recordings.

HOW DO WE KNOW WHICH IS CLOSER TO REALITY, WITH NO GROUND TRUTH?

There is no motion-capture reference here, so we cannot measure error directly.
Instead we test each reconstruction against facts that are true of the real world
no matter what any sensor reports. Reality has invariants; the reconstruction that
violates them less is the more accurate one. That is the whole basis of this
comparison.

  A. RIGID-SEGMENT CONSTANCY  <- the primary test
     A femur does not change length while you walk. Measure every bone on every
     frame and see how much its length wanders. Needs no external reference, and
     it is a hard physical constraint, so it is the strongest evidence available.

  B. BILATERAL SYMMETRY
     Left and right bones are near-equal in LENGTH. This holds even in
     pathological gait, which alters motion rather than skeletal proportions.

  C. ABSOLUTE PLAUSIBILITY
     A thigh is ~42 cm, a shin ~41 cm. Constancy alone is not enough -- a method
     could report a rock-steady but wrong length. A and C together catch both
     failure modes: precision and accuracy.

  D. FLOOR PLANARITY
     The subject walks on a flat floor, so ankles during stance must lie on a
     single plane. RMS deviation from the best-fit plane is an error measure.

  E. PATH STRAIGHTNESS / SCALE
     Context. The walk is roughly straight, but not perfectly, so this is weaker
     evidence than A-D and is reported for information only.

  F. SIGNAL QUALITY -- EXPLICITLY NOT AN ACCURACY MEASURE
     Jitter and impossible-velocity counts. Reported last and labelled, because
     SMOOTHNESS IS NOT ACCURACY: a constant output has zero jitter and infinite
     error. These numbers only mean something once A-D are comparable.

Reads the landmarks.npz written by run_examples.py, so it scores exactly the
artifacts that were shipped rather than re-tracking.

Usage:
    python compare.py                       # every example under ../Examples
    python compare.py <example folder> ...
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES = os.path.normpath(os.path.join(HERE, "..", "Examples"))

SEGMENTS = [
    ("L thigh", "left hip", "left knee", 42.0),
    ("R thigh", "right hip", "right knee", 42.0),
    ("L shin", "left knee", "left ankle", 41.0),
    ("R shin", "right knee", "right ankle", 41.0),
    ("L upper arm", "left shoulder", "left elbow", 30.0),
    ("R upper arm", "right shoulder", "right elbow", 30.0),
    ("L forearm", "left elbow", "left wrist", 26.0),
    ("R forearm", "right elbow", "right wrist", 26.0),
    ("shoulder width", "left shoulder", "right shoulder", None),
    ("hip width", "left hip", "right hip", None),
]
PAIRS = [("thigh", "L thigh", "R thigh"), ("shin", "L shin", "R shin"),
         ("upper arm", "L upper arm", "R upper arm"),
         ("forearm", "L forearm", "R forearm")]
BY_NAME = {s[0]: s for s in SEGMENTS}

# No joint exceeds this while walking; a foot in swing peaks near 4-5 m/s, so
# anything past 6 m/s is a reconstruction artefact rather than motion.
MAX_JOINT_SPEED = 6.0


def load(example, which):
    p = os.path.join(example, "output", which, "landmarks.npz")
    if not os.path.exists(p):
        return None
    z = np.load(p)
    out = {}
    for k in z.files:
        a = np.array(z[k], dtype=float)
        a[np.all(a == 0, axis=1)] = np.nan       # (0,0,0) marks an untracked frame
        out[k] = a
    return out


def seg_series(P, a, b):
    return np.linalg.norm(P[a] - P[b], axis=1) * 100.0


def robust_cv(v):
    """MAD-based coefficient of variation, %, immune to a handful of wild frames."""
    v = v[np.isfinite(v)]
    if v.size < 10:
        return np.nan
    med = np.median(v)
    if med == 0:
        return np.nan
    return 100.0 * (np.median(np.abs(v - med)) * 1.4826) / med


def plane_rms(pts):
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    if len(pts) < 20:
        return np.nan
    c = pts.mean(axis=0)
    n = np.linalg.svd(pts - c, full_matrices=False)[2][2]
    return float(np.sqrt(np.mean(((pts - c) @ n) ** 2)) * 100.0)


def line_rms(xy):
    xy = xy[np.all(np.isfinite(xy), axis=1)]
    if len(xy) < 20:
        return np.nan
    c = xy.mean(axis=0)
    n = np.linalg.svd(xy - c, full_matrices=False)[2][1]
    return float(np.sqrt(np.mean(((xy - c) @ n) ** 2)) * 100.0)


def analyse(P):
    r = {"segments": {}, "symmetry": {}}
    for nm, a, b, expect in SEGMENTS:
        v = seg_series(P, a, b)
        r["segments"][nm] = (float(np.nanmedian(v)), robust_cv(v), expect)
    for nm, ln, rn in PAIRS:
        lm = r["segments"][ln][0]
        rm = r["segments"][rn][0]
        r["symmetry"][nm] = 100.0 * abs(lm - rm) / np.mean([lm, rm])

    # D: floor plane from stance-phase ankles. +y is down in camera coords, so
    # the LOWER the foot the LARGER y; stance is the bottom 40% of heights.
    ank = []
    for side in ("left ankle", "right ankle"):
        A = P[side]
        ok = np.all(np.isfinite(A), axis=1)
        if ok.sum() > 20:
            thr = np.percentile(A[ok][:, 1], 60)
            ank.append(A[ok & (A[:, 1] >= thr)])
    r["floor_rms"] = plane_rms(np.vstack(ank)) if ank else np.nan

    hip = (P["left hip"] + P["right hip"]) / 2.0
    r["path_rms"] = line_rms(hip[:, [0, 2]])
    r["dist_med"] = float(np.nanmedian(hip[:, 2]) * 100.0)

    sp = []
    for nm in P:
        d = np.linalg.norm(np.diff(P[nm], axis=0), axis=1) * 60.0
        sp.append(d[np.isfinite(d)])
    allsp = np.concatenate(sp) if sp else np.array([np.nan])
    r["impossible_pct"] = float(100.0 * np.mean(allsp > MAX_JOINT_SPEED))
    r["speed_p50"] = float(np.median(allsp))
    r["speed_p99"] = float(np.percentile(allsp, 99))
    r["valid_pct"] = float(100.0 * np.mean(
        [np.mean(np.all(np.isfinite(P[k]), axis=1)) for k in P]))
    return r


def report(a, b, out=sys.stdout):
    def w(s=""):
        print(s, file=out)
    W = 30
    score = {"lidar": 0, "nolidar": 0}
    w("=" * 76)
    w(f"{'':<{W}}{'LiDAR':>13}{'no-LiDAR':>13}   winner")
    w("=" * 76)

    w("\n[A] RIGID-SEGMENT CONSTANCY - robust CV %, LOWER IS BETTER")
    w("    a bone cannot change length; this is the primary accuracy test")
    for nm, _, _, _ in SEGMENTS:
        ca, cb = a["segments"][nm][1], b["segments"][nm][1]
        win = "no-LiDAR" if cb < ca else "LiDAR"
        score["nolidar" if cb < ca else "lidar"] += 1
        w(f"  {nm:<{W-2}}{ca:>12.2f}%{cb:>12.2f}%   {win}")
    w(f"  {'-> segments won':<{W-2}}{score['lidar']:>13}{score['nolidar']:>13}")

    w("\n[B] BILATERAL SYMMETRY - |L-R|/mean %, LOWER IS BETTER")
    for nm, _, _ in PAIRS:
        va, vb = a["symmetry"][nm], b["symmetry"][nm]
        w(f"  {nm:<{W-2}}{va:>12.2f}%{vb:>12.2f}%   "
          f"{'no-LiDAR' if vb < va else 'LiDAR'}")

    w("\n[C] ABSOLUTE PLAUSIBILITY - median length vs expected anatomy")
    for nm, _, _, expect in SEGMENTS:
        if expect is None:
            continue
        ma, mb = a["segments"][nm][0], b["segments"][nm][0]
        ea, eb = abs(ma - expect), abs(mb - expect)
        w(f"  {nm:<{W-2}}{ma:>11.1f}cm{mb:>11.1f}cm   expect {expect:.0f}cm  "
          f"{'no-LiDAR' if eb < ea else 'LiDAR'}")

    w("\n[D] FLOOR PLANARITY - RMS of stance ankles from best-fit plane")
    fa, fb = a["floor_rms"], b["floor_rms"]
    w(f"  {'floor RMS':<{W-2}}{fa:>11.2f}cm{fb:>11.2f}cm   "
      f"{'no-LiDAR' if fb < fa else 'LiDAR'}")

    w("\n[E] PATH / SCALE - context, weaker evidence")
    w(f"  {'hip track RMS from line':<{W-2}}{a['path_rms']:>11.2f}cm"
      f"{b['path_rms']:>11.2f}cm")
    w(f"  {'median subject distance':<{W-2}}{a['dist_med']:>11.1f}cm"
      f"{b['dist_med']:>11.1f}cm   ratio {b['dist_med']/a['dist_med']:.3f}x")

    w("\n[F] SIGNAL QUALITY - NOT accuracy (a flat line would score perfectly)")
    w(f"  {'impossible speed >6 m/s':<{W-2}}{a['impossible_pct']:>12.2f}%"
      f"{b['impossible_pct']:>12.2f}%")
    w(f"  {'median joint speed m/s':<{W-2}}{a['speed_p50']:>13.3f}"
      f"{b['speed_p50']:>13.3f}")
    w(f"  {'99th pct joint speed m/s':<{W-2}}{a['speed_p99']:>13.3f}"
      f"{b['speed_p99']:>13.3f}")
    w(f"  {'frames with valid data':<{W-2}}{a['valid_pct']:>12.1f}%"
      f"{b['valid_pct']:>12.1f}%")
    w("=" * 76)
    return score


def find_examples():
    out = []
    for kind in ("torso", "hand"):
        d = os.path.join(EXAMPLES, kind)
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            p = os.path.join(d, name)
            if os.path.isdir(os.path.join(p, "output")):
                out.append(p)
    return out


if __name__ == "__main__":
    folders = sys.argv[1:] or find_examples()
    if not folders:
        raise SystemExit(f"No processed examples under {EXAMPLES}")
    tally = {"lidar": 0, "nolidar": 0}
    for f in folders:
        name = os.path.basename(os.path.normpath(f))
        L, M = load(f, "lidar"), load(f, "nolidar")
        if L is None or M is None:
            print(f"\n#### {name}: missing output for "
                  f"{'lidar' if L is None else 'nolidar'}, skipped")
            continue
        n = min(len(next(iter(L.values()))), len(next(iter(M.values()))))
        L = {k: v[:n] for k, v in L.items()}
        M = {k: v[:n] for k, v in M.items()}
        header = f"\n\n################ {name}  ({n} frames) ################"
        print(header)
        s = report(analyse(L), analyse(M))
        tally["lidar"] += s["lidar"]
        tally["nolidar"] += s["nolidar"]
        # keep a copy alongside the example's output
        dest = os.path.join(f, "output", "comparison.txt")
        with open(dest, "w", encoding="utf-8") as fh:
            fh.write(header.strip() + "\n")
            report(analyse(L), analyse(M), out=fh)
    print(f"\n\nOVERALL segment-constancy wins:  "
          f"LiDAR {tally['lidar']}   no-LiDAR {tally['nolidar']}")
