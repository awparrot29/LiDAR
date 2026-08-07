import os
import cv2
import numpy as np
import mediapipe as mp

import sessiongeom

# Map body part names to mp landmark numbers
landmark_map = {
    "left shoulder": 11,
    "right shoulder": 12,
    "left elbow": 13,
    "right elbow": 14,
    "left wrist": 15,
    "right wrist": 16,
    "left hip": 23,
    "right hip": 24,
    "left knee": 25,
    "right knee": 26,
    "left ankle": 27,
    "right ankle": 28,
}

# Default LiDAR frame dimensions (portrait). The real values come from
# sessiongeom.frame_geometry(), which accounts for how the device was held.
width, height = 192, 256


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

    # One output array per landmark
    arrays = {name: [] for name in landmark_map}
    prev = {name: None for name in landmark_map}

    # Pose estimation setup
    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(video_path)
    with mp_pose.Pose(static_image_mode=False, model_complexity=2,
                      enable_segmentation=False,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:

        frame_i = 0

        # Read frames
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame_i >= n_lidar:
                break

            # Stand the subject up (portrait recordings only) and match the
            # LiDAR resolution, then convert to RGB for mediapipe
            if rotate:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            frame = cv2.resize(frame, (width, height))
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = pose.process(rgb_frame)

            if frame_i == 0:
                print("Loading...")
            if frame_i % 10 == 0:
                # Progress marker consumed by the web app's job runner
                print(f"@@FRAME {frame_i}/{n_lidar}", flush=True)

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

            # Takes in a landmark name/index and returns the 3d coordinate of the point if it is confident enough, otherwise defaulting back to that of the previous frame
            def extract_one(name, idx):
                if results.pose_landmarks:
                    lm = results.pose_landmarks.landmark[idx]
                    pos = (int(lm.x * width), int(lm.y * height))
                    prev[name] = pos
                else:
                    pos = prev[name]

                if pos:
                    pointx = int(np.clip(pos[0], 0, width - 1))
                    pointy = int(np.clip(pos[1], 0, height - 1))
                    if frame_i == 0 or conf is None or conf[pointy, pointx] > 0:
                        z = depth_meters[pointy, pointx]
                    else:
                        # Fallback to previous z
                        z = arrays[name][frame_i - 1][2]
                    x = (pointx - cx) * z / fx
                    y = (pointy - cy) * z / fy
                    return (x, y, z)
                else:
                    print(f"No position for {name} at frame {frame_i}")
                    return (0, 0, 0)

            # Add the 3d coordinates to the arrays
            for name, idx in landmark_map.items():
                arrays[name].append(extract_one(name, idx))

            frame_i += 1

    cap.release()
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
