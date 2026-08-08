"""Blur the subject's face in a Stray Scanner recording.

A session zip contains rgb.mp4 at full resolution, which identifies whoever was
recorded. This rewrites that video with the head obscured and repacks the zip,
so a recording can be shared or published without exposing the participant. The
depth and confidence maps are 192x256 silhouettes carrying no facial detail and
are copied through untouched, as are the intrinsics and IMU files.

    python anonymize_session.py session.zip
    python anonymize_session.py session.zip -o shareable.zip --pixelate

The head is located from MediaPipe's pose landmarks rather than a face detector.
Face detectors lose a subject several metres away, which is most of a gait
recording, whereas the pose model tracks the whole body throughout and its
nose, eye, ear and mouth landmarks bound the head directly. When a frame gives
no pose at all the previous region is reused, enlarged, so no frame is left
unblurred.
"""
import argparse
import os
import shutil
import sys
import tempfile
import zipfile

import cv2
import numpy as np

# MediaPipe pose landmarks 0-10 are nose, eyes, ears and mouth
HEAD_LANDMARKS = list(range(11))

# The head box is grown by this fraction of its own size before blurring, so
# hair, chin and the edges of the face are covered even when a landmark is off.
PADDING = 1.5

# Floor on the blurred region, as a fraction of frame height. At five metres the
# head spans very few pixels and a box that small would leave the face legible.
MIN_SIZE_FRAC = 0.045


def head_box(landmarks, w, h):
    """Padded pixel box around the head, or None if the head is not visible."""
    pts = []
    for i in HEAD_LANDMARKS:
        lm = landmarks[i]
        # visibility is MediaPipe's own confidence for that point
        if lm.visibility > 0.25:
            pts.append((lm.x * w, lm.y * h))
    if len(pts) < 2:
        return None

    pts = np.array(pts)
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = max((x1 - x0), (y1 - y0)) * (1 + PADDING) / 2
    half = max(half, MIN_SIZE_FRAC * h / 2)
    return (int(cx - half), int(cy - half), int(cx + half), int(cy + half))


def obscure(frame, box, pixelate):
    """Blur or pixelate the box, feathered so the edit is not a hard rectangle."""
    h, w = frame.shape[:2]
    x0, y0 = max(0, box[0]), max(0, box[1])
    x1, y1 = min(w, box[2]), min(h, box[3])
    if x1 - x0 < 4 or y1 - y0 < 4:
        return frame

    roi = frame[y0:y1, x0:x1]
    if pixelate:
        blocks = 6
        small = cv2.resize(roi, (blocks, blocks), interpolation=cv2.INTER_AREA)
        hidden = cv2.resize(small, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
    else:
        k = max(31, (max(x1 - x0, y1 - y0) // 2) | 1)   # odd kernel, scales with box
        hidden = cv2.GaussianBlur(roi, (k, k), 0)

    # Elliptical mask so the result does not read as an obvious rectangle
    mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.ellipse(mask, ((x1 - x0) // 2, (y1 - y0) // 2),
                ((x1 - x0) // 2, (y1 - y0) // 2), 0, 0, 360, 255, -1)
    mask = cv2.GaussianBlur(mask, (31, 31), 0).astype(np.float32) / 255.0
    mask = mask[..., None]
    frame[y0:y1, x0:x1] = (hidden * mask + roi * (1 - mask)).astype(np.uint8)
    return frame


def open_writer(path, fps, w, h):
    """H.264 where available, so the result plays anywhere."""
    try:
        import imageio.v2 as iio
        return iio.get_writer(path, fps=fps, codec="libx264",
                              pixelformat="yuv420p", macro_block_size=1), "h264"
    except Exception:
        wr = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        if not wr.isOpened():
            raise SystemExit("Could not open a video writer")
        return wr, "mp4v"


def blur_video(src, dst, pixelate, quiet):
    import mediapipe as mp

    cap = cv2.VideoCapture(src)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer, kind = open_writer(dst, fps, w, h)
    last_box, blurred, missed = None, 0, 0

    with mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1,
                                min_detection_confidence=0.3,
                                min_tracking_confidence=0.3) as pose:
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            box = None
            if res.pose_landmarks:
                box = head_box(res.pose_landmarks.landmark, w, h)

            if box is None:
                missed += 1
                if last_box is not None:
                    # Widen the stale box, since the head has probably moved
                    x0, y0, x1, y1 = last_box
                    g = int(0.35 * max(x1 - x0, y1 - y0))
                    box = (x0 - g, y0 - g, x1 + g, y1 + g)
            else:
                last_box = box

            if box is not None:
                frame = obscure(frame, box, pixelate)
                blurred += 1

            if kind == "h264":
                writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                writer.write(frame)

            i += 1
            if not quiet and i % 20 == 0:
                pct = 100.0 * i / max(total, 1)
                sys.stdout.write(f"\r  blurring {i}/{total} ({pct:4.1f}%)")
                sys.stdout.flush()

    cap.release()
    writer.close() if kind == "h264" else writer.release()
    if not quiet:
        sys.stdout.write("\r" + " " * 50 + "\r")
    return i, blurred, missed, kind


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Blur the subject's face in a Stray Scanner session zip.")
    p.add_argument("zip", help="the session zip to anonymise")
    p.add_argument("-o", "--out", default=None,
                   help="output zip (default: <name>_anon.zip)")
    p.add_argument("--pixelate", action="store_true",
                   help="pixelate instead of blurring")
    p.add_argument("-q", "--quiet", action="store_true")
    args = p.parse_args(argv)

    if not zipfile.is_zipfile(args.zip):
        raise SystemExit(f"Not a zip file: {args.zip}")
    out_zip = args.out or os.path.splitext(args.zip)[0] + "_anon.zip"

    work = tempfile.mkdtemp(prefix="anon_")
    try:
        if not args.quiet:
            print(f"Unpacking {os.path.basename(args.zip)} ...")
        with zipfile.ZipFile(args.zip) as zf:
            names = zf.namelist()
            for info in zf.infolist():
                name = info.filename.replace("\\", "/")
                if name.endswith("/"):
                    continue
                parts = [q for q in name.split("/") if q not in ("", ".", "..")]
                dest = os.path.join(work, *parts)
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                with zf.open(info) as s, open(dest, "wb") as d:
                    shutil.copyfileobj(s, d)

        videos = []
        for root, _, files in os.walk(work):
            for f in files:
                if f.lower().endswith(".mp4"):
                    videos.append(os.path.join(root, f))
        if not videos:
            raise SystemExit("No .mp4 found in the zip")

        for path in videos:
            if not args.quiet:
                print(f"Blurring {os.path.relpath(path, work)} ...")
            tmp = path + ".anon.mp4"
            n, blurred, missed, kind = blur_video(path, tmp, args.pixelate,
                                                  args.quiet)
            os.replace(tmp, path)
            if not args.quiet:
                print(f"  {blurred}/{n} frames obscured, {missed} with no pose "
                      f"detected (covered from the previous frame), codec {kind}")

        if not args.quiet:
            print("Repacking ...")
        with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(work):
                for f in sorted(files):
                    full = os.path.join(root, f)
                    zf.write(full, os.path.relpath(full, work).replace("\\", "/"))

        if not args.quiet:
            mb = os.path.getsize(out_zip) / 1e6
            print(f"\nWrote {out_zip}  ({mb:.1f} MB, {len(names)} entries)")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
