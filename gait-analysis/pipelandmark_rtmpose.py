import os
import cv2
import numpy as np
from rtmlib import Body, PoseTracker

import background
import depthsmooth
import sessiongeom

# COCO 17-keypoint indices used by RTMPose
# (replaces MediaPipe's landmark numbering scheme)
landmark_map = {
    "left shoulder": 5,
    "right shoulder": 6,
    "left elbow": 7,
    "right elbow": 8,
    "left wrist": 9,
    "right wrist": 10,
    "left hip": 11,
    "right hip": 12,
    "left knee": 13,
    "right knee": 14,
    "left ankle": 15,
    "right ankle": 16,
}

# Default LiDAR frame dimensions (portrait). The real values come from
# sessiongeom.frame_geometry(), which accounts for how the device was held.
width, height = 192, 256

# Person detection is the expensive half of the pipeline, so reuse each box for
# the following frames. At 60 FPS the subject barely moves in 10 frames — this
# measured 0.4 px off full per-frame detection while running ~8x faster.
DET_FREQUENCY = 10


def _make_tracker():
    """RTMPose (rtmpose-m) on ONNX Runtime — no torch, no mmcv, no mmdet.

    Weights download once to ~/.cache/rtmlib (~155 MB: yolox_m + rtmpose-m).
    """
    return PoseTracker(Body, mode='balanced', backend='onnxruntime',
                       device='cpu', det_frequency=DET_FREQUENCY,
                       tracking=False, to_openpose=False)


def extract_all_landmarks(folder):
    """Track every landmark in one pass over the video.

    Returns {landmark name: [(x, y, z), ...]} in metres, one entry per frame.
    A single pass serves all joints — calculateangle used to re-run the whole
    video once per joint, which cost 6x more inference than necessary.
    """
    # Paths
    video_path = os.path.join(folder, "rgb.mp4")
    depth_folder = os.path.join(folder, "depth")
    conf_folder = os.path.join(folder, "confidence")

    # Ignore non-frame entries (.DS_Store, AppleDouble ._* files a Mac/iOS zip adds)
    def frame_pngs(d):
        return sorted(f for f in os.listdir(d)
                      if f.lower().endswith('.png') and not f.startswith('.'))

    # Depth drives the frame sequence; confidence is matched by filename rather
    # than by list position. A session with gaps in confidence/ would otherwise
    # silently pair depth 000000 with confidence 000026 and so on.
    depth_files = frame_pngs(depth_folder)
    conf_names = set(frame_pngs(conf_folder))

    n_lidar = len(depth_files)
    if n_lidar == 0:
        raise RuntimeError(f"No depth PNGs found in {depth_folder}")

    n_missing = sum(1 for f in depth_files if f not in conf_names)
    if n_missing:
        print(f"Warning: {n_missing}/{n_lidar} frames have no confidence map; "
              f"treating their depth as valid.")

    # How the device was held decides whether frames need rotating, and the
    # rotation swaps the axes of the camera matrix, so the two come together.
    geom = sessiongeom.frame_geometry(folder)
    width, height = geom['width'], geom['height']
    rotate = geom['rotate']
    fx, fy = geom['fx'], geom['fy']
    cx, cy = geom['cx'], geom['cy']
    print(f"Session recorded in {geom['orientation']}; "
          f"frame {width}x{height}, rotate={rotate}")

    # First pass records the pixel each joint landed on and the raw depth there;
    # the depths are then smoothed across neighbouring frames before being
    # projected, so the projection uses the settled distance rather than a
    # single noisy sample.
    pix = {name: [] for name in landmark_map}       # (col, row) per frame
    raw_z = {name: [] for name in landmark_map}     # metres, straight from LiDAR
    ok_z = {name: [] for name in landmark_map}      # was the reading usable
    prev = {name: None for name in landmark_map}

    # Learn the static scene so readings that are actually the wall or floor can
    # be thrown out before any averaging happens.
    bg_model = background.build_model(depth_folder, depth_files, rotate)
    drift = background.camera_drift(folder)
    if bg_model is None:
        print("No background model (too few depth frames); "
              "background rejection disabled")
    else:
        print(f"Background model built from "
              f"{min(len(depth_files), background.MAX_MODEL_FRAMES)} frames"
              + (f"; camera drift {drift*100:.1f} cm" if drift is not None else ""))
        if drift is not None and drift > background.DRIFT_WARN_M:
            print(f"Warning: camera moved {drift*100:.0f} cm during the recording, "
                  f"so the background model may be unreliable.")
    n_bg_rejected = {name: 0 for name in landmark_map}

    tracker = _make_tracker()
    cap = cv2.VideoCapture(video_path)
    frame_i = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_i >= n_lidar:
            break

        # Stand the subject up (portrait recordings only) and match the
        # LiDAR resolution
        if rotate:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        frame = cv2.resize(frame, (width, height))

        if frame_i == 0:
            print("Loading...")
        if frame_i % 10 == 0:
            # Progress marker consumed by the web app's job runner
            print(f"@@FRAME {frame_i}/{n_lidar}", flush=True)

        # Run RTMPose — keypoints (N, 17, 2), scores (N, 17)
        keypoints, scores = tracker(frame)
        kps = None
        if keypoints is not None and len(keypoints):
            # Pick the most confident person (handles multi-person scenes)
            best = int(np.argmax(np.asarray(scores).mean(axis=1)))
            kps = np.asarray(keypoints)[best]

        # Load depth/confidence once per frame, pairing them by filename
        fname = depth_files[frame_i]
        depth_mm = cv2.imread(os.path.join(depth_folder, fname), cv2.IMREAD_UNCHANGED)
        if depth_mm is None:
            print(f"Missing depth at frame {frame_i} ({fname})")
            break

        # conf stays None when this frame has no confidence map — the depth
        # is then taken at face value.
        conf = None
        if fname in conf_names:
            conf = cv2.imread(os.path.join(conf_folder, fname), cv2.IMREAD_UNCHANGED)

        # Depth and confidence must be turned the same way as the RGB frame
        if rotate:
            depth_mm = cv2.rotate(depth_mm, cv2.ROTATE_90_CLOCKWISE)
            if conf is not None:
                conf = cv2.rotate(conf, cv2.ROTATE_90_CLOCKWISE)
        depth_meters = depth_mm / 1000.0

        # Record where each landmark sits and what the LiDAR reads there.
        # Nothing is projected yet - that waits until the depths have been
        # smoothed across frames.
        def sample_one(name, idx):
            if kps is not None:
                pos = (int(kps[idx][0]), int(kps[idx][1]))
                prev[name] = pos
            else:
                pos = prev[name]

            if not pos:
                print(f"No position for {name} at frame {frame_i}")
                return (0, 0), 0.0, False

            pointx = int(np.clip(pos[0], 0, width - 1))
            pointy = int(np.clip(pos[1], 0, height - 1))
            z = float(depth_meters[pointy, pointx])
            # An unusable reading is excluded from the average rather than
            # replaced by the previous frame, so it cannot drag its neighbours;
            # smoothing fills the gap from both sides.
            usable = z > 0 and (conf is None or conf[pointy, pointx] > 0)
            # The landmark can sit on a limb the depth map failed to register,
            # in which case the pixel reports the scene behind it.
            if usable and background.is_background(bg_model, z, pointx, pointy):
                n_bg_rejected[name] += 1
                usable = False
            return (pointx, pointy), z, usable

        for name, idx in landmark_map.items():
            p, z, usable = sample_one(name, idx)
            pix[name].append(p)
            raw_z[name].append(z)
            ok_z[name].append(usable)

        frame_i += 1

    cap.release()

    # Second pass: average each joint's depth over the neighbouring frames, then
    # project. x and y are derived from the smoothed depth so all three stay
    # consistent with one another.
    print(f"Smoothing depth over {depthsmooth.DEFAULT_WINDOW} frames "
          f"({frame_i} frames tracked)")
    total_rejected = sum(n_bg_rejected.values())
    if total_rejected:
        worst = max(n_bg_rejected.items(), key=lambda kv: kv[1])
        print(f"Rejected {total_rejected} readings that were the background "
              f"({100*total_rejected/max(frame_i*len(landmark_map),1):.1f}% of samples; "
              f"most affected: {worst[0]}, {worst[1]})")

    arrays = {}
    for name in landmark_map:
        z = depthsmooth.smooth_series(raw_z[name], ok_z[name],
                                      depthsmooth.DEFAULT_WINDOW)
        pts = []
        for (pointx, pointy), zi in zip(pix[name], z):
            if zi <= 0:
                pts.append((0, 0, 0))
                continue
            pts.append(((pointx - cx) * zi / fx,
                        (pointy - cy) * zi / fy,
                        zi))
        arrays[name] = pts

    return arrays


def extract_landmarks(folder, input1, input2, input3):
    """Back-compatible wrapper returning just the three requested landmarks."""
    arrays = extract_all_landmarks(folder)
    l1_arr, l2_arr, l3_arr = arrays[input1], arrays[input2], arrays[input3]

    # Save to cache
    save_dir = os.path.join(folder, input1)
    os.makedirs(save_dir, exist_ok=True)
    np.savez(os.path.join(save_dir, "landmark_cache.npz"),
             l1=l1_arr, l2=l2_arr, l3=l3_arr,
             input1=input1, input2=input2, input3=input3)

    return l1_arr, l2_arr, l3_arr, input1, input2, input3


if __name__ == '__main__':
    folder = input("Folder name: ")
    input1 = input("left knee, right knee, left elbow, or right elbow: ")
    input2 = input("left hip, right hip, left shoulder, or right shoulder: ")
    input3 = input("left ankle, right ankle, left wrist, or right wrist: ")
    extract_landmarks(folder, input1, input2, input3)
