"""Decide whether a recording shows a torso or a close-up hand.

Measured on this project's four recordings, the discriminator is not subtle:

    recording          nearest object   lower-body visibility   hand found
    hand close-up            203 mm             0.00               100%
    park_sim                1517 mm             0.99                45%
    Jul_30                  1360 mm             0.99                36%
    Test3 (iPhone)          1850 mm             0.99                55%

Lower-body visibility separates them completely, 0.00 against 0.99, with nothing
in between. Nearest-object distance corroborates.

Note that hand-detection rate is a POOR signal on its own — MediaPipe finds a hand
on the walking subject 36-55% of the time — so it is used only as a tie-breaker.

The caller can always override this; auto-detection is a convenience, not a
constraint, and `explain()` returns the evidence so a wrong call is visible rather
than silent.
"""
import glob
import os

import cv2
import mediapipe as mp
import numpy as np

import profiles
import sessiongeom

# Lower-body visibility above this means a body is in frame. The observed gap is
# 0.00 vs 0.99, so the exact value is not delicate.
VISIBILITY_TORSO = 0.5

# A subject this close is a close-up, not someone walking. Observed 203 mm for the
# hand and >= 1360 mm for every torso recording.
NEAR_HAND_MM = 700.0

N_SAMPLE = 10
_LOWER = [mp.solutions.pose.PoseLandmark.LEFT_HIP.value,
          mp.solutions.pose.PoseLandmark.RIGHT_HIP.value,
          mp.solutions.pose.PoseLandmark.LEFT_KNEE.value,
          mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value,
          mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value,
          mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value]


def _evidence(folder, n_sample=N_SAMPLE):
    geom = sessiongeom.frame_geometry(folder)
    dw, dh, rot = geom["width"], geom["height"], geom["rotate"]

    dp = sorted(f for f in glob.glob(os.path.join(folder, "depth", "*.png"))
                if not os.path.basename(f).startswith("."))
    idx = np.linspace(0, max(len(dp) - 1, 0), min(n_sample, max(len(dp), 1)))
    idx = np.unique(idx.astype(int))

    near = []
    for i in idx:
        if i < len(dp):
            im = cv2.imread(dp[i], cv2.IMREAD_UNCHANGED)
            if im is None:
                continue
            v = im.astype(np.float32)
            v = v[v > 0]
            if v.size:
                near.append(float(np.percentile(v, 1)))

    cap = cv2.VideoCapture(os.path.join(folder, "rgb.mp4"))
    vis, hand_hits, n = [], 0, 0
    pose = mp.solutions.pose.Pose(static_image_mode=True, model_complexity=1,
                                  min_detection_confidence=0.5)
    hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1,
                                     model_complexity=1,
                                     min_detection_confidence=0.5)
    try:
        for i in idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, frame = cap.read()
            if not ok:
                continue
            if rot:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            frame = cv2.resize(frame, (dw, dh))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            n += 1
            pr = pose.process(rgb)
            if pr.pose_landmarks:
                lm = pr.pose_landmarks.landmark
                vis.append(float(np.median([lm[k].visibility for k in _LOWER])))
            else:
                vis.append(0.0)
            if hands.process(rgb).multi_hand_landmarks:
                hand_hits += 1
    finally:
        pose.close()
        hands.close()
        cap.release()

    return {
        "nearest_mm": float(np.median(near)) if near else float("nan"),
        "lower_body_visibility": float(np.median(vis)) if vis else 0.0,
        "hand_rate": 100.0 * hand_hits / max(n, 1),
        "frames_sampled": n,
    }


def detect(folder, n_sample=N_SAMPLE):
    """Return (kind, evidence dict). kind is profiles.TORSO or profiles.HAND."""
    ev = _evidence(folder, n_sample)
    body_visible = ev["lower_body_visibility"] >= VISIBILITY_TORSO
    very_close = (np.isfinite(ev["nearest_mm"])
                  and ev["nearest_mm"] < NEAR_HAND_MM)

    if body_visible and not very_close:
        kind = profiles.TORSO
    elif not body_visible:
        kind = profiles.HAND
    else:
        # a body IS visible yet the subject is very close — trust the hand rate
        kind = profiles.HAND if ev["hand_rate"] >= 80 else profiles.TORSO
    ev["kind"] = kind
    return kind, ev


def explain(ev):
    """One-line summary of the evidence, so a wrong call is visible."""
    return (f"detected '{ev['kind']}' — lower-body visibility "
            f"{ev['lower_body_visibility']:.2f} "
            f"(torso if >= {VISIBILITY_TORSO}), nearest object "
            f"{ev['nearest_mm']:.0f} mm (hand if < {NEAR_HAND_MM:.0f}), "
            f"hand found {ev['hand_rate']:.0f}% of "
            f"{ev['frames_sampled']} sampled frames")
