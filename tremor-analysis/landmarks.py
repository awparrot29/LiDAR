"""Extract joint locations from a Stray Scanner recording — hand or whole body.

Produces, per frame:
  px    (n, 2)  landmark position in the working image, pixels
  world (n, 3)  the model's metric 3D, metres, origin at the hand/hip centre
  cam   (n, 3)  absolute metric 3D in camera coordinates, via solvePnP
  z0            distance to the subject, metres

No depth sensor is used. `depth/` and `confidence/` are never read, so this runs
on any iPhone or iPad. (Measured on this project's recordings: LiDAR depth is
~10x noisier than the 2D track inside the tremor band, and for a close-range hand
it was wrong by 2.4-3.7x on two of three recordings. See the README.)

WHY solvePnP RATHER THAN THE DEPTH MAP
MediaPipe returns the subject's 3D shape in real units but centred on the subject,
not positioned in the room. solvePnP recovers where that shape sits in front of
the camera from the 21 (or 33) 2D/3D correspondences, which is rigorous geometry
rather than a learned estimate. Scale ambiguity is broken by the model knowing how
big a hand or body is — anchor it to a ruler measurement via `scale` for a
specific subject.
"""
import glob
import os

import cv2
import mediapipe as mp
import numpy as np

import sessiongeom

HAND = "hand"
BODY = "body"

# MediaPipe hand landmark names, in index order
HAND_NAMES = [
    "wrist",
    "thumb CMC", "thumb MCP", "thumb IP", "thumb tip",
    "index MCP", "index PIP", "index DIP", "index tip",
    "middle MCP", "middle PIP", "middle DIP", "middle tip",
    "ring MCP", "ring PIP", "ring DIP", "ring tip",
    "pinky MCP", "pinky PIP", "pinky DIP", "pinky tip",
]

# Body landmarks we keep. The full 33 include face points that are not useful
# here; these are the joints the gait pipeline already tracks plus the feet.
BODY_KEEP = {
    "left shoulder": 11, "right shoulder": 12,
    "left elbow": 13, "right elbow": 14,
    "left wrist": 15, "right wrist": 16,
    "left hip": 23, "right hip": 24,
    "left knee": 25, "right knee": 26,
    "left ankle": 27, "right ankle": 28,
    "left heel": 29, "right heel": 30,
    "left foot": 31, "right foot": 32,
}

# Spans that cannot change length, used as a self-check on the reconstruction.
HAND_RIGID = [(0, 9, "wrist->midMCP"), (5, 17, "idxMCP->pinkyMCP"),
              (5, 9, "idxMCP->midMCP"), (9, 13, "midMCP->ringMCP"),
              (13, 17, "ringMCP->pinkyMCP")]

# Long side of the working frame. The 2D track is the primary tremor signal, so
# unlike the gait pipeline there is no reason to shrink to the depth resolution;
# more pixels means a finer noise floor.
DEFAULT_LONG_SIDE = 640

HAND_CONNECTIONS = list(mp.solutions.hands.HAND_CONNECTIONS)


def _geometry(folder, long_side=DEFAULT_LONG_SIDE):
    """Working frame and intrinsics, derived without needing depth/."""
    orientation = sessiongeom.session_orientation(folder)
    cap = cv2.VideoCapture(os.path.join(folder, "rgb.mp4"))
    nw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    nh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1440
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    cap.release()

    scale = long_side / float(max(nw, nh))
    ww, wh = int(round(nw * scale)), int(round(nh * scale))

    M = np.loadtxt(os.path.join(folder, "camera_matrix.csv"), delimiter=",")
    f = float(M[0, 0]) * scale
    cx, cy = float(M[0, 2]) * scale, float(M[1, 2]) * scale
    fx = fy = f

    if orientation == sessiongeom.PORTRAIT:
        # ROTATE_90_CLOCKWISE maps (u, v) -> (H - 1 - v, u): the axes swap
        width, height = wh, ww
        fx, fy = fy, fx
        cx, cy = (wh - 1) - cy, cx
        rotate = True
    else:
        width, height = ww, wh
        rotate = False

    return {"orientation": orientation, "rotate": rotate,
            "width": width, "height": height, "fps": float(fps),
            "fx": fx, "fy": fy, "cx": cx, "cy": cy, "n_video": total}


def camera_motion(folder):
    """Total device path length and max displacement, in metres.

    Camera shake appears as apparent joint tremor, so a recording where the
    device moved is not trustworthy for tremor. Returns (path, max_disp) or None.
    """
    p = os.path.join(folder, "odometry.csv")
    if not os.path.exists(p):
        return None
    rows = []
    with open(p, "r", errors="ignore") as fh:
        fh.readline()
        for line in fh:
            q = line.strip().split(",")
            if len(q) >= 5:
                try:
                    rows.append([float(q[2]), float(q[3]), float(q[4])])
                except ValueError:
                    continue
    if len(rows) < 2:
        return None
    a = np.array(rows)
    step = np.linalg.norm(np.diff(a, axis=0), axis=1)
    return float(step.sum()), float(np.linalg.norm(a - a[0], axis=1).max())


def extract(folder, kind=HAND, n_frames=None, scale=1.0, long_side=DEFAULT_LONG_SIDE,
            max_hands=1, progress=True):
    """Track a session.

    scale multiplies the model's metric output — set it from a ruler measurement
    of this subject (see README) so absolute amplitudes are right for them.

    Returns a dict with 'names', 'px', 'world', 'cam', 'z0', 'geom', plus
    'connections' for rendering and 'rigid' spans for the self-check.
    """
    geom = _geometry(folder, long_side)
    W, H = geom["width"], geom["height"]
    K = np.array([[geom["fx"], 0, geom["cx"]],
                  [0, geom["fy"], geom["cy"]],
                  [0, 0, 1]], dtype=np.float64)

    if n_frames is None:
        depth_dir = os.path.join(folder, "depth")
        if os.path.isdir(depth_dir):
            # match the LiDAR pipeline's frame count when the folder happens to
            # be there, so results line up with existing output
            n_frames = len([f for f in os.listdir(depth_dir)
                            if f.lower().endswith(".png") and not f.startswith(".")])
        else:
            n_frames = geom["n_video"]

    if kind == HAND:
        names = HAND_NAMES
        keep = list(range(21))
        solver = mp.solutions.hands.Hands(
            static_image_mode=False, max_num_hands=max_hands, model_complexity=1,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)
        conns = HAND_CONNECTIONS
        rigid = HAND_RIGID
    else:
        names = list(BODY_KEEP)
        keep = [BODY_KEEP[k] for k in names]
        solver = mp.solutions.pose.Pose(
            static_image_mode=False, model_complexity=2,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)
        conns = None
        rigid = None

    n = len(keep)
    px = np.full((n_frames, n, 2), np.nan)
    world = np.full((n_frames, n, 3), np.nan)
    cam = np.full((n_frames, n, 3), np.nan)
    z0 = np.full(n_frames, np.nan)

    cap = cv2.VideoCapture(os.path.join(folder, "rgb.mp4"))
    i = 0
    with solver as s:
        while i < n_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if geom["rotate"]:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            frame = cv2.resize(frame, (W, H))
            res = s.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if progress and i % 30 == 0:
                print(f"@@FRAME {i}/{n_frames}", flush=True)

            if kind == HAND:
                got = bool(res.multi_hand_landmarks and res.multi_hand_world_landmarks)
                lm = res.multi_hand_landmarks[0].landmark if got else None
                wl = res.multi_hand_world_landmarks[0].landmark if got else None
            else:
                got = bool(res.pose_landmarks and res.pose_world_landmarks)
                lm = res.pose_landmarks.landmark if got else None
                wl = res.pose_world_landmarks.landmark if got else None

            if got:
                allpx = np.array([[l.x * W, l.y * H] for l in lm], float)
                allw = np.array([[l.x, l.y, l.z] for l in wl], float) * scale
                px[i] = allpx[keep]
                world[i] = allw[keep]
                ok2, rvec, tvec = cv2.solvePnP(allw[keep], allpx[keep], K, None,
                                               flags=cv2.SOLVEPNP_ITERATIVE)
                if ok2:
                    R, _ = cv2.Rodrigues(rvec)
                    cam[i] = (R @ allw[keep].T).T + tvec.ravel()
                    z0[i] = float(tvec.ravel()[2])
            i += 1
    cap.release()

    detected = int(np.isfinite(px[:, 0, 0]).sum())
    print(f"{kind}: {detected}/{i} frames tracked "
          f"({100.0 * detected / max(i, 1):.1f}%)")

    return {"kind": kind, "names": names, "px": px[:i], "world": world[:i],
            "cam": cam[:i], "z0": z0[:i], "geom": geom,
            "connections": conns, "rigid": rigid, "scale": scale}


def mm_per_pixel(track):
    """Lateral millimetres subtended by one pixel, at the subject's distance.

    Converts the 2D pixel track into millimetres. The 2D track is the primary
    tremor signal because it involves no depth estimate at all.
    """
    z = np.nanmedian(track["z0"])
    if not np.isfinite(z):
        return np.nan
    return 1000.0 * float(z) / track["geom"]["fx"]


def rigid_span_report(track):
    """Median length and frame-to-frame variability of spans that cannot change.

    A sanity check on the reconstruction: if these wander, the 3D is not
    trustworthy. Returns [(label, median_cm, robust_cv_pct), ...].
    """
    if not track["rigid"]:
        return []
    out = []
    for a, b, label in track["rigid"]:
        v = np.linalg.norm(track["cam"][:, a] - track["cam"][:, b], axis=1) * 100
        v = v[np.isfinite(v)]
        if v.size < 10:
            continue
        med = float(np.median(v))
        cv = 100.0 * float(np.median(np.abs(v - med)) * 1.4826) / med if med else np.nan
        out.append((label, med, cv))
    return out
