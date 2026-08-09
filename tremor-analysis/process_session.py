"""Map joint locations and quantify shaking, from one Stray Scanner recording.

    python process_session.py <session folder or .zip>
    python process_session.py session.zip --kind body
    python process_session.py hand_session --reference wrist --scale 1.02
    python process_session.py hand_session --no-render

Writes, under <out>/:
    data/<joint>.csv     per joint: x_px, y_px, x_mm, y_mm, z_mm per frame
    tremor.csv           per joint per axis: amplitude, frequency, SNR, constancy
    tremor_report.txt    the same, readable, with the self-checks
    landmarks.npz        raw arrays for re-analysis without re-tracking
    skeleton.mp4         two-view 3D skeleton (unless --no-render)

No depth sensor is used, so this runs on any iPhone or iPad.
"""
import argparse
import os
import shutil
import sys
import tempfile
import zipfile

import numpy as np

import landmarks
import tremor

HERE = os.path.dirname(os.path.abspath(__file__))


def _extract_zip(path, dest):
    """Extract, tolerating Windows-style backslash entry names and skipping the
    __MACOSX shadow tree, then return the folder holding the recording."""
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
        if "rgb.mp4" in files and "camera_matrix.csv" in files:
            return dirpath
    raise SystemExit("No Stray Scanner recording found inside the zip")


def write_outputs(track, res, mmpx, out_dir, session_label, cam_motion):
    data_dir = os.path.join(out_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    px_mm = track["px"] * mmpx
    cam_mm = track["cam"] * 1000.0
    for i, name in enumerate(track["names"]):
        rows = np.column_stack([track["px"][:, i, 0], track["px"][:, i, 1],
                                px_mm[:, i, 0], px_mm[:, i, 1], cam_mm[:, i, 2]])
        np.savetxt(os.path.join(data_dir, f"{name}.csv"), rows, delimiter=",",
                   header="x_px,y_px,x_mm,y_mm,z_mm", comments="", fmt="%.4f")
    np.savez(os.path.join(out_dir, "landmarks.npz"),
             px=track["px"], world=track["world"], cam=track["cam"],
             z0=track["z0"], names=np.array(track["names"]))

    # machine-readable summary
    cols = ["joint", "axis", "amp_rms_mm", "amp_pp_mm", "noise_rms_mm", "snr",
            "detected", "contaminated", "voluntary_ratio", "dom_freq_hz",
            "peak_sharpness", "constancy", "voluntary_rms_mm"]
    lines = [",".join(cols)]
    for joint, axes in res["joints"].items():
        for key in ("in_plane_2d", "depth_3d"):
            if key not in axes:
                continue
            d = axes[key]
            lines.append(",".join([joint, d["axis"]] + [
                f"{d[c]:.4f}" if isinstance(d[c], float) else str(d[c])
                for c in cols[2:]]))
    with open(os.path.join(out_dir, "tremor.csv"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    # human-readable report
    L = []
    a = L.append
    a(f"TREMOR REPORT — {session_label}")
    a("=" * 78)
    g = track["geom"]
    a(f"frames {res['n_frames']}   {g['fps']:.0f} fps   "
      f"{res['n_frames'] / g['fps']:.1f} s   working frame "
      f"{g['width']}x{g['height']}   {g['orientation']}")
    a(f"subject distance {np.nanmedian(track['z0']) * 1000:.0f} mm   "
      f"1 px = {mmpx:.2f} mm laterally   model scale factor {track['scale']:.4f}")
    a(f"frequency resolution {res['freq_resolution_hz']:.3f} Hz   "
      f"reference joint: {res['reference'] or 'none (absolute motion)'}")
    a("")
    a(f"bands: voluntary {tremor.VOLUNTARY_BAND[0]}-{tremor.VOLUNTARY_BAND[1]} Hz | "
      f"tremor {tremor.TREMOR_BAND[0]}-{tremor.TREMOR_BAND[1]} Hz | "
      f"noise {tremor.NOISE_BAND[0]}-{tremor.NOISE_BAND[1]} Hz")
    a(f"a joint counts as tremulous at SNR >= {tremor.SNR_THRESHOLD:.0f} "
      f"against its own noise band")
    a("")

    a("-- CHECKS ------------------------------------------------------------")
    if cam_motion:
        path, disp = cam_motion
        verdict = "OK" if disp < 0.01 else "WARNING: camera moved"
        a(f"camera motion: path {path * 100:.1f} cm, max displacement "
          f"{disp * 100:.2f} cm  -> {verdict}")
        if disp >= 0.01:
            a("   camera shake is indistinguishable from real tremor in the")
            a("   absolute measurement; treat amplitudes with suspicion.")
    else:
        a("camera motion: no odometry.csv, could not check")
    spans = landmarks.rigid_span_report(track)
    if spans:
        a("rigid spans (cannot change length — a self-check on the 3D):")
        for label, med, cv in spans:
            flag = "" if cv < 10 else "   <- high, 3D may be unreliable"
            a(f"   {label:22} {med:6.2f} cm   variability {cv:5.1f}%{flag}")
    det = int(np.isfinite(track['px'][:, 0, 0]).sum())
    a(f"tracking: {det}/{res['n_frames']} frames "
      f"({100.0 * det / max(res['n_frames'], 1):.1f}%)")
    a("")

    a("-- IN-PLANE (2D image plane — the trustworthy axis) ------------------")
    a(f"{'joint':14}{'amp p-p':>10}{'noise':>9}{'SNR':>7}{'freq':>8}"
      f"{'vol/trem':>10}{'const':>8}  tremor?")
    n_contam = 0
    for joint, axes in res["joints"].items():
        d = axes.get("in_plane_2d")
        if not d:
            continue
        if d["contaminated"]:
            n_contam += 1
            verdict = "masked by movement"
        else:
            verdict = "YES" if d["detected"] else "-"
        a(f"{joint:14}{d['amp_pp_mm']:8.2f}mm{d['noise_rms_mm']:7.2f}mm"
          f"{d['snr']:7.1f}{d['dom_freq_hz']:7.2f}Hz{d['voluntary_ratio']:10.1f}"
          f"{100 * d['constancy']:7.0f}%   {verdict}")
    if n_contam:
        a("")
        a(f"WARNING  {n_contam} joints are dominated by voluntary movement "
          f"(vol/trem > {tremor.VOLUNTARY_DOMINANCE:.0f}).")
        a("         Voluntary motion is not sinusoidal, so a movement task throws")
        a("         harmonics into the 3-8 Hz band and mimics tremor. Assess")
        a("         tremor on a recording of the limb AT REST, not during a task.")
    a("")
    a("-- DEPTH AXIS (~10x noisier — for reference only) --------------------")
    a(f"{'joint':14}{'amp p-p':>10}{'noise':>9}{'SNR':>7}")
    for joint, axes in res["joints"].items():
        d = axes.get("depth_3d")
        if not d:
            continue
        a(f"{joint:14}{d['amp_pp_mm']:8.2f}mm{d['noise_rms_mm']:7.2f}mm"
          f"{d['snr']:7.1f}")
    a("")
    a("NOTE  amp p-p is peak-to-peak, converted from band RMS assuming a")
    a("      sinusoid. 'noise' is that joint's own 12-25 Hz amplitude, so any")
    a("      figure with SNR below ~3 is not distinguishable from noise.")
    a("=" * 78)
    text = "\n".join(L)
    with open(os.path.join(out_dir, "tremor_report.txt"), "w",
              encoding="utf-8") as fh:
        fh.write(text + "\n")
    return text


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("session", help="session folder or .zip")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--kind", choices=(landmarks.HAND, landmarks.BODY),
                    default=landmarks.HAND)
    ap.add_argument("--reference", default=None,
                    help="joint to subtract first, e.g. 'wrist' to isolate "
                         "finger tremor from whole-hand movement")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="per-subject scale for the model's metric output, from "
                         "a ruler measurement (see README)")
    ap.add_argument("--long-side", type=int, default=landmarks.DEFAULT_LONG_SIDE)
    ap.add_argument("--max-hands", type=int, default=1)
    ap.add_argument("--no-render", action="store_true")
    args = ap.parse_args(argv)

    tmp = None
    session = args.session
    if os.path.isfile(session) and zipfile.is_zipfile(session):
        tmp = tempfile.mkdtemp(prefix="tremor_")
        session = _extract_zip(session, tmp)
    if not os.path.isdir(session):
        raise SystemExit(f"Not a session folder or zip: {args.session}")

    label = os.path.basename(os.path.normpath(args.session))
    out_dir = args.out or os.path.join(
        os.path.dirname(os.path.abspath(os.path.normpath(session))),
        os.path.basename(os.path.normpath(session)), "output", "tremor")
    os.makedirs(out_dir, exist_ok=True)

    try:
        cam_motion = landmarks.camera_motion(session)
        track = landmarks.extract(session, kind=args.kind, scale=args.scale,
                                  long_side=args.long_side,
                                  max_hands=args.max_hands)
        mmpx = landmarks.mm_per_pixel(track)
        ref = None
        if args.reference:
            if args.reference not in track["names"]:
                raise SystemExit(f"Unknown reference joint '{args.reference}'. "
                                 f"Choices: {', '.join(track['names'])}")
            ref = track["names"].index(args.reference)
        res = tremor.analyse_track(track, mmpx, reference=ref)
        text = write_outputs(track, res, mmpx, out_dir, label, cam_motion)
        print("\n" + text)

        if not args.no_render and track["connections"]:
            try:
                import render
                path = os.path.join(out_dir, "skeleton.mp4")
                render.two_view_video(track, path)
                print(f"\nwrote {path}")
            except Exception as exc:
                print(f"\nrender skipped ({type(exc).__name__}: {exc})")
        print(f"\nOutput: {out_dir}")
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
