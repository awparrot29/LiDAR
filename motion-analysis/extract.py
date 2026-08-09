"""Joint coordinates from a Stray Scanner recording — one engine, either subject.

This is the gait pipeline's pipelandmark.py generalised: the only thing the subject
kind changes is which MediaPipe model runs and which landmarks are kept, both of
which come from profiles.py. Everything else is identical for a torso and a hand:

  - device orientation from imu.csv, so portrait and landscape both come out
    upright (sessiongeom)
  - RGB resized to the depth resolution, so a landmark pixel indexes the depth map
    directly with no registration step
  - depth read at each landmark, rejected when the sensor's confidence is zero or
    when the reading is really the background (background)
  - each joint's depth averaged over neighbouring frames with outliers dropped
    (depthsmooth) before projecting, so the projection uses a settled distance
  - back-projection x = (col - cx) * z / fx, y = (row - cy) * z / fy

Returns {landmark name: [(x, y, z), ...]} in metres, one entry per frame, with an
untracked frame stored as (0, 0, 0) — the same convention the gait pipeline uses.

WHICH RECORDINGS WORK
Two of three hand recordings taken for this project were silently defective: the
confidence channel was zero everywhere and depth read 2.4-3.7x too far, while
looking perfect to the eye because the crisp outline comes from the RGB image. The
confidence gate is what catches that; it rejected every sample from both. Watch the
"usable depth" percentage printed below before trusting a new recording.
"""
import os

import cv2
import mediapipe as mp
import numpy as np

import background
import depthsmooth
import profiles
import sessiongeom


def _frame_pngs(d):
    """Frame files only — a Mac/iOS zip adds .DS_Store and AppleDouble ._* files."""
    return sorted(f for f in os.listdir(d)
                  if f.lower().endswith('.png') and not f.startswith('.'))


def _open_model(profile, max_hands):
    if profile["model"] == "hands":
        return mp.solutions.hands.Hands(
            static_image_mode=False, max_num_hands=max_hands,
            model_complexity=1, min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
    return mp.solutions.pose.Pose(
        static_image_mode=False, model_complexity=2, enable_segmentation=False,
        min_detection_confidence=0.5, min_tracking_confidence=0.5)


def _landmarks_of(profile, results):
    """The per-frame landmark list, or None when nothing was detected."""
    if profile["model"] == "hands":
        if results.multi_hand_landmarks:
            return results.multi_hand_landmarks[0].landmark
        return None
    if results.pose_landmarks:
        return results.pose_landmarks.landmark
    return None


def extract_all_landmarks(folder, kind=profiles.TORSO, max_hands=1):
    """Track every landmark of the given subject kind in one pass over the video."""
    profile = profiles.get(kind)
    landmark_map = profile["landmarks"]

    video_path = os.path.join(folder, "rgb.mp4")
    depth_folder = os.path.join(folder, "depth")
    conf_folder = os.path.join(folder, "confidence")

    depth_files = _frame_pngs(depth_folder)
    conf_names = set(_frame_pngs(conf_folder)) if os.path.isdir(conf_folder) else set()

    n_lidar = len(depth_files)
    if n_lidar == 0:
        raise RuntimeError(f"No depth PNGs found in {depth_folder}")

    # Confidence is matched by FILENAME, not list position. A session with gaps in
    # confidence/ would otherwise pair depth 000000 with confidence 000026.
    n_missing = sum(1 for f in depth_files if f not in conf_names)
    if n_missing:
        print(f"Warning: {n_missing}/{n_lidar} frames have no confidence map; "
              f"treating their depth as valid.")

    geom = sessiongeom.frame_geometry(folder)
    width, height = geom['width'], geom['height']
    rotate = geom['rotate']
    fx, fy = geom['fx'], geom['fy']
    cx, cy = geom['cx'], geom['cy']
    print(f"Subject: {kind}.  Session recorded in {geom['orientation']}; "
          f"frame {width}x{height}, rotate={rotate}")

    pix = {name: [] for name in landmark_map}      # (col, row) per frame
    raw_z = {name: [] for name in landmark_map}    # metres, straight from LiDAR
    ok_z = {name: [] for name in landmark_map}     # was the reading usable
    prev = {name: None for name in landmark_map}

    use_bg = profile.get("background_rejection", True)
    bg_model = (background.build_model(depth_folder, depth_files, rotate)
                if use_bg else None)
    drift = background.camera_drift(folder)
    if not use_bg:
        print("Background rejection disabled for this subject kind "
              "(a stationary close-up would be learned as background)")
    elif bg_model is None:
        print("No background model (too few depth frames); "
              "background rejection disabled")
    else:
        print(f"Background model built from "
              f"{min(len(depth_files), background.MAX_MODEL_FRAMES)} frames"
              + (f"; camera drift {drift*100:.1f} cm" if drift is not None else ""))
        if drift is not None and drift > background.DRIFT_WARN_M:
            print(f"Warning: camera moved {drift*100:.0f} cm during the "
                  f"recording, so the background model may be unreliable.")
    n_bg_rejected = {name: 0 for name in landmark_map}
    n_undetected = 0

    cap = cv2.VideoCapture(video_path)
    with _open_model(profile, max_hands) as model:
        frame_i = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame_i >= n_lidar:
                break

            if rotate:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            frame = cv2.resize(frame, (width, height))
            results = model.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if frame_i == 0:
                print("Loading...")
            if frame_i % 10 == 0:
                # Progress marker consumed by the web app's job runner
                print(f"@@FRAME {frame_i}/{n_lidar}", flush=True)

            fname = depth_files[frame_i]
            depth_mm = cv2.imread(os.path.join(depth_folder, fname),
                                  cv2.IMREAD_UNCHANGED)
            if depth_mm is None:
                print(f"Missing depth at frame {frame_i} ({fname})")
                break

            conf = None
            if fname in conf_names:
                conf = cv2.imread(os.path.join(conf_folder, fname),
                                  cv2.IMREAD_UNCHANGED)

            # Depth and confidence must be turned the same way as the RGB frame
            if rotate:
                depth_mm = cv2.rotate(depth_mm, cv2.ROTATE_90_CLOCKWISE)
                if conf is not None:
                    conf = cv2.rotate(conf, cv2.ROTATE_90_CLOCKWISE)
            depth_meters = depth_mm / 1000.0

            lms = _landmarks_of(profile, results)
            if lms is None:
                n_undetected += 1

            for name, idx in landmark_map.items():
                if lms is not None:
                    lm = lms[idx]
                    pos = (int(lm.x * width), int(lm.y * height))
                    prev[name] = pos
                else:
                    pos = prev[name]

                if not pos:
                    pix[name].append((0, 0))
                    raw_z[name].append(0.0)
                    ok_z[name].append(False)
                    continue

                pointx = int(np.clip(pos[0], 0, width - 1))
                pointy = int(np.clip(pos[1], 0, height - 1))
                z = float(depth_meters[pointy, pointx])
                # An unusable reading is excluded from the average rather than
                # replaced by the previous frame, so it cannot drag its
                # neighbours; smoothing fills the gap from both sides.
                usable = z > 0 and (conf is None or conf[pointy, pointx] > 0)
                # A limb or fingertip is thin, so the pixel can report the scene
                # behind it instead of the joint.
                if usable and background.is_background(bg_model, z, pointx, pointy):
                    n_bg_rejected[name] += 1
                    usable = False
                pix[name].append((pointx, pointy))
                raw_z[name].append(z)
                ok_z[name].append(usable)

            frame_i += 1
    cap.release()

    if n_undetected:
        print(f"Warning: no {kind} detected on {n_undetected}/{frame_i} frames "
              f"(held previous position)")

    n_samples = frame_i * len(landmark_map)
    n_usable = sum(sum(1 for v in ok_z[n] if v) for n in landmark_map)
    pct = 100.0 * n_usable / max(n_samples, 1)
    print(f"Usable depth: {n_usable}/{n_samples} samples ({pct:.1f}%)")
    if n_usable == 0:
        print("ERROR: not one depth sample passed the confidence gate. This "
              "recording is almost certainly defective — re-record and check "
              "that the confidence maps are not all zero.")
    elif pct < 20:
        print(f"Warning: only {pct:.1f}% of depth samples were usable; treat the "
              f"coordinates with caution.")

    total_rejected = sum(n_bg_rejected.values())
    if total_rejected:
        worst = max(n_bg_rejected.items(), key=lambda kv: kv[1])
        print(f"Rejected {total_rejected} readings that were the background "
              f"({100.0*total_rejected/max(n_samples,1):.1f}% of samples; "
              f"most affected: {worst[0]}, {worst[1]})")

    # Anatomical plausibility: the subject has a bounded depth extent, so a
    # reading far outside it is the scene behind the joint rather than the joint.
    # Done before smoothing so a bad sample never enters the average.
    max_extent = profile.get("max_depth_extent_m")
    n_implausible = 0
    if max_extent:
        names = list(landmark_map)
        for i in range(frame_i):
            zs = [raw_z[n][i] for n in names if ok_z[n][i]]
            if len(zs) < 3:
                continue
            centre = float(np.median(zs))
            for n in names:
                if ok_z[n][i] and abs(raw_z[n][i] - centre) > max_extent:
                    ok_z[n][i] = False
                    n_implausible += 1
        if n_implausible:
            print(f"Rejected {n_implausible} readings more than "
                  f"{max_extent*100:.0f} cm in depth from the rest of the "
                  f"subject ({100.0*n_implausible/max(n_samples,1):.1f}% of samples)")

    print(f"Smoothing depth over {depthsmooth.DEFAULT_WINDOW} frames")
    arrays = {}
    for name in landmark_map:
        z = depthsmooth.smooth_series(raw_z[name], ok_z[name],
                                      depthsmooth.DEFAULT_WINDOW)
        pts = []
        for (pointx, pointy), zi in zip(pix[name], z):
            if zi is None or not np.isfinite(zi) or zi <= 0:
                pts.append((0, 0, 0))
                continue
            pts.append(((pointx - cx) * zi / fx,
                        (pointy - cy) * zi / fy,
                        zi))
        arrays[name] = pts
    return arrays, geom
