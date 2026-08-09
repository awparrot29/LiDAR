"""No-LiDAR joint tracking: 3D body landmarks from RGB video alone.

A clone of LiDAR-Gait-Analysis/gait-analysis/pipelandmark.py with the depth
sensor removed. Same public interface, same output format
({landmark name: [(x, y, z), ...]} in metres), so downstream code
(calculateangle.py, skeleton3d.py) works unchanged.

`depth/` and `confidence/` are never read. Nothing here requires a LiDAR-equipped
device, so it runs on any iPhone or iPad.

WHAT CHANGED, AND WHAT DELIBERATELY DID NOT
Only the SOURCE OF Z. The 2D landmarks and the back-projection
    x = (col - cx) * z / fx      y = (row - cy) * z / fy
are byte-for-byte the same operation as the LiDAR pipeline. That is on purpose:
it makes the two a controlled comparison rather than two different programs, so
any difference in the output is attributable to depth alone.

WHERE Z COMES FROM
  1. `pose_world_landmarks` places every joint in metres relative to the midpoint
     of the hips, supplying the body's SHAPE including relative depth. This is a
     learned anatomical model, not a measurement.
  2. That is hip-relative and needs anchoring to an absolute distance, recovered
     from apparent torso size by similar triangles:
         z0 = physical_span * f / pixel_span
     taken as the median over six torso spans. Validated against LiDAR on
     park_sim: 0.99x, correlation 0.984 across the walk.
  3. z_joint = z0 + world_z_joint  (world z is 0 at the hip midpoint and
     increases away from the camera).

NO TEMPORAL SMOOTHING is applied. The LiDAR path needs its depthsmooth stage
because individual depth samples are noisy. Whether this path needs one is an
open question that compare.py measures rather than assumes.

LIMITATION TO BE AWARE OF
Depth here is a model estimate, so it inherits the model's priors. It is expected
to be weakest where the body is foreshortened along the view axis, and it has
not been validated on pathological gait.
"""
import os

import cv2
import mediapipe as mp
import numpy as np

import sessiongeom

# Mirrors landmark_map in the LiDAR pipeline's pipelandmark.py. Kept local so
# this folder stands alone; if the original ever changes, update both.
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

_P = mp.solutions.pose.PoseLandmark

# Torso spans used to anchor absolute distance: the torso is the most nearly
# rigid part of the body and the least often self-occluded during walking.
_TORSO_SPANS = [
    (_P.LEFT_SHOULDER.value, _P.LEFT_HIP.value),
    (_P.RIGHT_SHOULDER.value, _P.RIGHT_HIP.value),
    (_P.LEFT_SHOULDER.value, _P.RIGHT_HIP.value),
    (_P.RIGHT_SHOULDER.value, _P.LEFT_HIP.value),
    (_P.LEFT_SHOULDER.value, _P.RIGHT_SHOULDER.value),
    (_P.LEFT_HIP.value, _P.RIGHT_HIP.value),
]

# Long side of the working frame when there is no depth/ folder to match.
DEFAULT_LONG_SIDE = 256

# A span must project to at least this many pixels to anchor distance; a nearly
# edge-on torso span collapses toward zero and would blow z0 up.
MIN_SPAN_PX = 5.0


def _frame_pngs(d):
    return sorted(f for f in os.listdir(d)
                  if f.lower().endswith('.png') and not f.startswith('.'))


def frame_geometry_no_depth(folder):
    """sessiongeom.frame_geometry equivalent that does not need depth/.

    frame_geometry probes a depth PNG to fix the working resolution, so it cannot
    run on a session recorded without LiDAR. This reproduces it from rgb.mp4,
    camera_matrix.csv and imu.csv only.
    """
    orientation = sessiongeom.session_orientation(folder)

    cap = cv2.VideoCapture(os.path.join(folder, "rgb.mp4"))
    native_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    native_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1440
    cap.release()

    scale = DEFAULT_LONG_SIDE / float(max(native_w, native_h))
    work_w = int(round(native_w * scale))
    work_h = int(round(native_h * scale))

    matrix = np.loadtxt(os.path.join(folder, "camera_matrix.csv"), delimiter=",")
    f_native = float(matrix[0, 0])
    cx_native, cy_native = float(matrix[0, 2]), float(matrix[1, 2])
    fx = fy = f_native * scale
    cx, cy = cx_native * scale, cy_native * scale

    if orientation == sessiongeom.PORTRAIT:
        # cv2.ROTATE_90_CLOCKWISE maps (u, v) -> (H - 1 - v, u): the axes swap
        width, height = work_h, work_w
        fx, fy = fy, fx
        cx, cy = (work_h - 1) - cy, cx
        rotate = True
    else:
        width, height = work_w, work_h
        rotate = False

    return {"orientation": orientation, "rotate": rotate,
            "width": width, "height": height,
            "fx": fx, "fy": fy, "cx": cx, "cy": cy,
            "native": (native_w, native_h), "depth": None}


def resolve_geometry(folder, match_lidar_frame=True):
    """Working frame and intrinsics.

    match_lidar_frame reuses the LiDAR pipeline's frame when depth/ happens to be
    present. That is what makes an A/B comparison valid — identical working
    resolution means both pipelines see identical input frames and therefore
    identical 2D landmarks. Set it False to run as a device without LiDAR would.
    """
    depth_dir = os.path.join(folder, "depth")
    if match_lidar_frame and os.path.isdir(depth_dir) and _frame_pngs(depth_dir):
        return sessiongeom.frame_geometry(folder)
    return frame_geometry_no_depth(folder)


def _distance_to_hips(world, px, fx, fy):
    """Absolute distance in metres to the hip midpoint, from apparent torso size.

    Returns None when no torso span projects to a usable pixel length.
    """
    est = []
    for a, b in _TORSO_SPANS:
        dx = px[a][0] - px[b][0]
        dy = px[a][1] - px[b][1]
        pix = float(np.hypot(dx, dy))
        if pix < MIN_SPAN_PX:
            continue
        phys = float(np.linalg.norm(world[a] - world[b]))
        if phys <= 0:
            continue
        # Use the focal length of the axis the span mostly runs along so a
        # non-square working frame does not bias vertical against horizontal.
        f = fx if abs(dx) >= abs(dy) else fy
        est.append(phys * f / pix)
    return float(np.median(est)) if est else None


def extract_all_landmarks(folder, n_frames=None, match_lidar_frame=True):
    """Track every landmark in one pass, taking depth from the body model.

    n_frames bounds the run, so it can be held to the same frame count as a LiDAR
    run on the same session. Defaults to the depth frame count when depth/ is
    present (for comparability), otherwise the whole video.
    """
    video_path = os.path.join(folder, "rgb.mp4")

    geom = resolve_geometry(folder, match_lidar_frame)
    width, height = geom['width'], geom['height']
    rotate = geom['rotate']
    fx, fy = geom['fx'], geom['fy']
    cx, cy = geom['cx'], geom['cy']

    if n_frames is None:
        depth_dir = os.path.join(folder, "depth")
        if match_lidar_frame and os.path.isdir(depth_dir):
            n_frames = len(_frame_pngs(depth_dir))
        else:
            cap0 = cv2.VideoCapture(video_path)
            n_frames = int(cap0.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            cap0.release()

    src = "matched to depth/" if geom['depth'] else "derived (no LiDAR needed)"
    print(f"Session recorded in {geom['orientation']}; "
          f"frame {width}x{height}, rotate={rotate}; geometry {src}")
    print("Depth source: MediaPipe metric body model (no depth sensor used)")

    arrays = {name: [] for name in landmark_map}
    prev_pix = {name: None for name in landmark_map}
    prev_world = None
    prev_z0 = None
    n_no_pose = 0
    n_no_dist = 0

    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(video_path)
    with mp_pose.Pose(static_image_mode=False, model_complexity=2,
                      enable_segmentation=False,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:
        frame_i = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame_i >= n_frames:
                break

            if rotate:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            frame = cv2.resize(frame, (width, height))
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if frame_i == 0:
                print("Loading...")
            if frame_i % 10 == 0:
                # Progress marker consumed by the web app's job runner
                print(f"@@FRAME {frame_i}/{n_frames}", flush=True)

            has = bool(results.pose_landmarks and results.pose_world_landmarks)
            if has:
                lms = results.pose_landmarks.landmark
                px = [(l.x * width, l.y * height) for l in lms]
                world = np.array([[l.x, l.y, l.z]
                                  for l in results.pose_world_landmarks.landmark])
                z0 = _distance_to_hips(world, px, fx, fy)
                if z0 is None:
                    n_no_dist += 1
                    z0 = prev_z0
                else:
                    prev_z0 = z0
                prev_world = world
            else:
                n_no_pose += 1
                world = prev_world
                z0 = prev_z0

            for name, idx in landmark_map.items():
                if has:
                    pos = (int(px[idx][0]), int(px[idx][1]))
                    prev_pix[name] = pos
                else:
                    pos = prev_pix[name]

                if pos is None or z0 is None or world is None:
                    arrays[name].append((0, 0, 0))
                    continue

                pointx = int(np.clip(pos[0], 0, width - 1))
                pointy = int(np.clip(pos[1], 0, height - 1))
                zi = z0 + float(world[idx][2])
                if zi <= 0:
                    arrays[name].append((0, 0, 0))
                    continue
                arrays[name].append(((pointx - cx) * zi / fx,
                                     (pointy - cy) * zi / fy,
                                     zi))
            frame_i += 1
    cap.release()

    if n_no_pose:
        print(f"Warning: no pose on {n_no_pose}/{frame_i} frames "
              f"(held previous position)")
    if n_no_dist:
        print(f"Warning: torso too foreshortened to anchor distance on "
              f"{n_no_dist}/{frame_i} frames (held previous distance)")
    print(f"Tracked {frame_i} frames without a depth sensor")
    return arrays


def extract_landmarks(folder, input1, input2, input3):
    """Back-compatible wrapper matching pipelandmark.extract_landmarks."""
    arrays = extract_all_landmarks(folder)
    l1_arr, l2_arr, l3_arr = arrays[input1], arrays[input2], arrays[input3]
    save_dir = os.path.join(folder, input1)
    os.makedirs(save_dir, exist_ok=True)
    np.savez(os.path.join(save_dir, "landmark_cache_nolidar.npz"),
             l1=l1_arr, l2=l2_arr, l3=l3_arr,
             input1=input1, input2=input2, input3=input3)
    return l1_arr, l2_arr, l3_arr, input1, input2, input3


if __name__ == '__main__':
    import sys
    for f in sys.argv[1:]:
        a = extract_all_landmarks(f)
        n = len(next(iter(a.values())))
        print(f"{f}: {n} frames, {len(a)} landmarks")
