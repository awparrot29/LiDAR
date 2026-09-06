"""What differs between a torso, hand, or foot recording — and nothing else.

The pipelines share the whole engine: orientation detection, depth sampling,
confidence and background rejection, depth smoothing, back-projection, CSV output
and rendering. The only things that actually differ are which model to run, which
landmarks to keep, which bones to draw, and which angles to compute. Those live
here as declarative tables so there is exactly one engine to maintain.

Feet in the TORSO profile are deliberately absent. Measured on this project's
recordings, the heel and toe landmarks land only 5.5-7.6 px apart in forward-walking
footage, giving a foot length of 8-10 cm against a real ~25 cm. The FOOT profile
addresses this via RTMPose-WholeBody133, which adds dedicated foot keypoints. A
transverse walk (Phase 3) is still recommended to eliminate foreshortening.
"""
import mediapipe as mp

TORSO = "torso"
HAND = "hand"
FOOT = "foot"

_P = mp.solutions.pose.PoseLandmark

# --------------------------------------------------------------------------
# Torso — matches the gait pipeline's landmark_map exactly, so output from this
# program is interchangeable with output from LiDAR-Gait-Analysis.
# --------------------------------------------------------------------------
TORSO_LANDMARKS = {
    "left shoulder": _P.LEFT_SHOULDER.value,
    "right shoulder": _P.RIGHT_SHOULDER.value,
    "left elbow": _P.LEFT_ELBOW.value,
    "right elbow": _P.RIGHT_ELBOW.value,
    "left wrist": _P.LEFT_WRIST.value,
    "right wrist": _P.RIGHT_WRIST.value,
    "left hip": _P.LEFT_HIP.value,
    "right hip": _P.RIGHT_HIP.value,
    "left knee": _P.LEFT_KNEE.value,
    "right knee": _P.RIGHT_KNEE.value,
    "left ankle": _P.LEFT_ANKLE.value,
    "right ankle": _P.RIGHT_ANKLE.value,
}

TORSO_BONES = [
    ("left shoulder", "right shoulder"), ("left hip", "right hip"),
    ("left shoulder", "left hip"), ("right shoulder", "right hip"),
    ("left shoulder", "left elbow"), ("left elbow", "left wrist"),
    ("right shoulder", "right elbow"), ("right elbow", "right wrist"),
    ("left hip", "left knee"), ("left knee", "left ankle"),
    ("right hip", "right knee"), ("right knee", "right ankle"),
]

# pivot -> (pivot, arm A, arm C); the angle is measured at the pivot.
# Same six the gait pipeline produces.
TORSO_ANGLES = {
    "left elbow": ("left elbow", "left shoulder", "left wrist"),
    "right elbow": ("right elbow", "right shoulder", "right wrist"),
    "left hip": ("left hip", "left shoulder", "left knee"),
    "right hip": ("right hip", "right shoulder", "right knee"),
    "left knee": ("left knee", "left hip", "left ankle"),
    "right knee": ("right knee", "right hip", "right ankle"),
}

# Joints whose z distance is worth graphing
TORSO_TRACES = ["left knee", "right knee", "left wrist", "right wrist"]

# --------------------------------------------------------------------------
# Hand — MediaPipe Hands returns these 21 in index order.
# --------------------------------------------------------------------------
HAND_LANDMARKS = {
    "wrist": 0,
    "thumb CMC": 1, "thumb MCP": 2, "thumb IP": 3, "thumb tip": 4,
    "index MCP": 5, "index PIP": 6, "index DIP": 7, "index tip": 8,
    "middle MCP": 9, "middle PIP": 10, "middle DIP": 11, "middle tip": 12,
    "ring MCP": 13, "ring PIP": 14, "ring DIP": 15, "ring tip": 16,
    "pinky MCP": 17, "pinky PIP": 18, "pinky DIP": 19, "pinky tip": 20,
}

HAND_BONES = [
    ("wrist", "thumb CMC"), ("thumb CMC", "thumb MCP"),
    ("thumb MCP", "thumb IP"), ("thumb IP", "thumb tip"),
    ("wrist", "index MCP"), ("index MCP", "index PIP"),
    ("index PIP", "index DIP"), ("index DIP", "index tip"),
    ("index MCP", "middle MCP"), ("middle MCP", "middle PIP"),
    ("middle PIP", "middle DIP"), ("middle DIP", "middle tip"),
    ("middle MCP", "ring MCP"), ("ring MCP", "ring PIP"),
    ("ring PIP", "ring DIP"), ("ring DIP", "ring tip"),
    ("ring MCP", "pinky MCP"), ("wrist", "pinky MCP"),
    ("pinky MCP", "pinky PIP"), ("pinky PIP", "pinky DIP"),
    ("pinky DIP", "pinky tip"),
]

# Three flexion angles per finger, down the chain. 180 degrees is straight.
HAND_ANGLES = {
    "thumb CMC": ("thumb CMC", "wrist", "thumb MCP"),
    "thumb MCP": ("thumb MCP", "thumb CMC", "thumb IP"),
    "thumb IP": ("thumb IP", "thumb MCP", "thumb tip"),
    "index MCP": ("index MCP", "wrist", "index PIP"),
    "index PIP": ("index PIP", "index MCP", "index DIP"),
    "index DIP": ("index DIP", "index PIP", "index tip"),
    "middle MCP": ("middle MCP", "wrist", "middle PIP"),
    "middle PIP": ("middle PIP", "middle MCP", "middle DIP"),
    "middle DIP": ("middle DIP", "middle PIP", "middle tip"),
    "ring MCP": ("ring MCP", "wrist", "ring PIP"),
    "ring PIP": ("ring PIP", "ring MCP", "ring DIP"),
    "ring DIP": ("ring DIP", "ring PIP", "ring tip"),
    "pinky MCP": ("pinky MCP", "wrist", "pinky PIP"),
    "pinky PIP": ("pinky PIP", "pinky MCP", "pinky DIP"),
    "pinky DIP": ("pinky DIP", "pinky PIP", "pinky tip"),
}

HAND_TRACES = ["wrist", "thumb tip", "index tip", "middle tip", "ring tip",
               "pinky tip"]

# --------------------------------------------------------------------------
# Toe Tap — RTMPose-WholeBody133, lower leg only.
#
# Tracks only knee + foot landmarks for MDS-UPDRS 3.7 (Toe Tapping).
# Dropping everything above the knee removes shoulder/hip/arm jitter from
# the visualization and focuses depth sampling on the small region that
# actually moves during the test.  Knee stays because the ankle
# dorsiflexion angle (knee→ankle→big_toe) is the primary clinical metric.
#
# Uses COCO-WholeBody keypoint indices:
#   Knees (13-14), ankles (15-16), foot (17-22)
# --------------------------------------------------------------------------
TOE_TAP = "toe_tap"

TOE_TAP_LANDMARKS = {
    "left knee": 13,    "right knee": 14,
    "left ankle": 15,   "right ankle": 16,
    "left big toe": 17, "left small toe": 18, "left heel": 19,
    "right big toe": 20, "right small toe": 21, "right heel": 22,
}

TOE_TAP_BONES = [
    ("left knee", "left ankle"),    ("right knee", "right ankle"),
    ("left heel", "left ankle"),
    ("left ankle", "left big toe"), ("left ankle", "left small toe"),
    ("left big toe", "left small toe"),
    ("right heel", "right ankle"),
    ("right ankle", "right big toe"), ("right ankle", "right small toe"),
    ("right big toe", "right small toe"),
]

TOE_TAP_ANGLES = {
    "left ankle dorsiflexion":  ("left ankle",  "left knee",  "left big toe"),
    "right ankle dorsiflexion": ("right ankle", "right knee", "right big toe"),
}

TOE_TAP_TRACES = ["left big toe", "right big toe", "left heel", "right heel"]

# --------------------------------------------------------------------------
# Foot — RTMPose-WholeBody133 via rtmlib.
#
# Uses COCO-WholeBody keypoint indices (NOT MediaPipe's numbering):
#   Body (0-16):  nose, eyes, ears, shoulders(5-6), elbows(7-8),
#                 wrists(9-10), hips(11-12), knees(13-14), ankles(15-16)
#   Foot (17-22): left big toe(17), left small toe(18), left heel(19),
#                 right big toe(20), right small toe(21), right heel(22)
#
# For test 3.8 (Leg Agility) — full body context needed.
# --------------------------------------------------------------------------
FOOT_LANDMARKS = {
    "left shoulder": 5, "right shoulder": 6,
    "left elbow": 7,    "right elbow": 8,
    "left wrist": 9,    "right wrist": 10,
    "left hip": 11,     "right hip": 12,
    "left knee": 13,    "right knee": 14,
    "left ankle": 15,   "right ankle": 16,
    "left big toe": 17, "left small toe": 18, "left heel": 19,
    "right big toe": 20, "right small toe": 21, "right heel": 22,
}

FOOT_BONES = [
    ("left shoulder", "right shoulder"), ("left hip", "right hip"),
    ("left shoulder", "left hip"),       ("right shoulder", "right hip"),
    ("left shoulder", "left elbow"),     ("left elbow", "left wrist"),
    ("right shoulder", "right elbow"),   ("right elbow", "right wrist"),
    ("left hip", "left knee"),           ("left knee", "left ankle"),
    ("right hip", "right knee"),         ("right knee", "right ankle"),
    # foot segments
    ("left heel", "left ankle"),
    ("left ankle", "left big toe"),   ("left ankle", "left small toe"),
    ("left big toe", "left small toe"),
    ("right heel", "right ankle"),
    ("right ankle", "right big toe"), ("right ankle", "right small toe"),
    ("right big toe", "right small toe"),
]

# pivot -> (pivot, arm_a, arm_c); angle at pivot.
# Includes the six standard torso angles plus ankle dorsiflexion/plantarflexion.
# "left ankle dorsiflexion": angle at left ankle between shin (knee→ankle) and
# foot (ankle→big_toe). ~90° = neutral, decreasing = plantarflexion (toe lift).
FOOT_ANGLES = {
    "left elbow":  ("left elbow",  "left shoulder",  "left wrist"),
    "right elbow": ("right elbow", "right shoulder", "right wrist"),
    "left hip":    ("left hip",    "left shoulder",  "left knee"),
    "right hip":   ("right hip",   "right shoulder", "right knee"),
    "left knee":   ("left knee",   "left hip",       "left ankle"),
    "right knee":  ("right knee",  "right hip",      "right ankle"),
    "left ankle dorsiflexion":  ("left ankle",  "left knee",  "left big toe"),
    "right ankle dorsiflexion": ("right ankle", "right knee", "right big toe"),
}

FOOT_TRACES = ["left knee", "right knee", "left big toe", "right big toe",
               "left heel", "right heel"]

# --------------------------------------------------------------------------
# Spine — SpinePose 37-keypoint model via spinepose library.
#
# Uses SpinePose keypoint indices (different from COCO/MediaPipe):
#   Body (0-16):   same as COCO-17 (shoulders 5-6, hips 11-12, knees 13-14)
#   Extra body (17-19): head(17), neck(18), hip midpoint(19)
#   Foot (20-25):  big-toe, small-toe, heel per side
#   Spine (26-30): spine_01(26) lumbar → spine_05(30) upper-thoracic
#   Other (31-36): latissimus, clavicle, neck_02, neck_03
#
# Spine chain bottom→top: hip(19)→26→27→28→29→30→neck(18)→35→36→head(17)
#
# For test 3.13 (Posture): measures kyphosis at each vertebral level and
# head/neck flexion to quantify Parkinsonian stooped posture.
# --------------------------------------------------------------------------
SPINE = "spine"

SPINE_LANDMARKS = {
    "left shoulder":  5,
    "right shoulder": 6,
    "left hip":       11,
    "right hip":      12,
    "left knee":      13,
    "right knee":     14,
    "hip":            19,
    "spine 01":       26,
    "spine 02":       27,
    "spine 03":       28,
    "spine 04":       29,
    "spine 05":       30,
    "neck":           18,
    "head":           17,
}

SPINE_BONES = [
    # Body frame for context
    ("left shoulder", "right shoulder"),
    ("left hip", "right hip"),
    ("left shoulder", "left hip"),
    ("right shoulder", "right hip"),
    ("left hip", "left knee"),
    ("right hip", "right knee"),
    # Connect midline to bilateral body
    ("hip", "left hip"), ("hip", "right hip"),
    ("spine 05", "left shoulder"), ("spine 05", "right shoulder"),
    # Spine chain (bottom to top)
    ("hip", "spine 01"), ("spine 01", "spine 02"),
    ("spine 02", "spine 03"), ("spine 03", "spine 04"),
    ("spine 04", "spine 05"), ("spine 05", "neck"),
    ("neck", "head"),
]

# pivot → (pivot, arm_a, arm_c); 180° = upright, decreasing = forward flexion.
# Each angle captures one vertebral section's contribution to stooped posture.
SPINE_ANGLES = {
    "cervical":       ("neck",     "head",     "spine 05"),
    "upper thoracic": ("spine 05", "neck",     "spine 04"),
    "mid thoracic":   ("spine 04", "spine 05", "spine 03"),
    "lower thoracic": ("spine 03", "spine 04", "spine 02"),
    "thoracolumbar":  ("spine 02", "spine 03", "spine 01"),
    "lumbar":         ("spine 01", "spine 02", "hip"),
}

SPINE_TRACES = ["head", "neck", "spine 03", "hip"]


REST_TREMOR = "rest_tremor"


def get(kind):
    """Profile for a subject kind: TORSO, HAND, FOOT, TOE_TAP, REST_TREMOR, or SPINE."""
    if kind == REST_TREMOR:
        return {"kind": REST_TREMOR, "landmarks": TORSO_LANDMARKS,
                "bones": TORSO_BONES, "angles": TORSO_ANGLES,
                "traces": TORSO_TRACES, "model": "pose",
                "background_rejection": True,
                "max_depth_extent_m": 1.5,
                # 3-frame pixel window (~50ms at 60fps) preserves 4-6 Hz rest
                # tremor (3.17a-d) that the default 9-frame window would cut.
                "pixel_smooth_window": 3}
    if kind == TOE_TAP:
        return {"kind": TOE_TAP, "landmarks": TOE_TAP_LANDMARKS,
                "bones": TOE_TAP_BONES, "angles": TOE_TAP_ANGLES,
                "traces": TOE_TAP_TRACES, "model": "wholebody",
                "background_rejection": True,
                # Lower leg spans ~50 cm in depth at most; tighter than full body.
                "max_depth_extent_m": 0.8,
                # 13-frame centred window (217 ms at 60 fps) passes 2-4 Hz taps
                # while suppressing RTMPose 2D jitter (~14-35 mm at 1 m).
                "pixel_smooth_window": 13}
    if kind == TORSO:
        return {"kind": TORSO, "landmarks": TORSO_LANDMARKS,
                "bones": TORSO_BONES, "angles": TORSO_ANGLES,
                "traces": TORSO_TRACES, "model": "pose",
                "background_rejection": True,
                # A walking person's limbs genuinely span a lot of depth (an arm
                # reaching forward while a leg trails), so be permissive here and
                # let background rejection do the work.
                "max_depth_extent_m": 1.5}
    if kind == HAND:
        return {"kind": HAND, "landmarks": HAND_LANDMARKS,
                "bones": HAND_BONES, "angles": HAND_ANGLES,
                "traces": HAND_TRACES, "model": "hands",
                # OFF for hands, and this is not a tuning preference.
                # background.build_model learns the persistent depth at each
                # pixel, which identifies the static scene only when the subject
                # moves THROUGH it. A hand held in a close-up stays put, so it
                # becomes the persistent content and the model learns the hand
                # itself as background — measured on the good recording, that
                # rejected 40.9% of samples including all 449 wrist readings and
                # cut usable depth from ~94% to 38.4%. The confidence gate still
                # protects against bad depth.
                "background_rejection": False,
                # A hand is only ~20 cm across, so no joint can sit 25 cm in depth
                # away from the rest of it. This replaces background rejection for
                # hands: a fingertip that samples the wall behind it lands far
                # outside the hand's own depth extent and is caught here. Observed
                # symptom without it — a single fingertip flung metres away,
                # dragging one bone across the whole stick figure.
                "max_depth_extent_m": 0.25,
                # 3-frame pixel window (~50ms at 60fps, cutoff ~9Hz) removes
                # single-frame landmark jitter without attenuating the clinical
                # signals: finger tapping (3.4-3.6, 2-5 Hz), postural tremor
                # (3.15, 4-8 Hz), and kinetic tremor (3.16, 4-8 Hz).
                "pixel_smooth_window": 3}
    if kind == FOOT:
        return {"kind": FOOT, "landmarks": FOOT_LANDMARKS,
                "bones": FOOT_BONES, "angles": FOOT_ANGLES,
                "traces": FOOT_TRACES, "model": "wholebody",
                "background_rejection": True,
                "max_depth_extent_m": 1.5,
                # Toe-tapping movements are 2-4 Hz; use a 7-frame pixel window
                # (117 ms at 60 fps) to smooth 2D jitter while preserving taps.
                "pixel_smooth_window": 7}
    if kind == SPINE:
        return {"kind": SPINE, "landmarks": SPINE_LANDMARKS,
                "bones": SPINE_BONES, "angles": SPINE_ANGLES,
                "traces": SPINE_TRACES, "model": "spine",
                "background_rejection": True,
                "max_depth_extent_m": 1.5}
    raise ValueError(
        f"Unknown kind '{kind}'. Use '{TORSO}', '{HAND}', '{FOOT}', "
        f"'{TOE_TAP}', '{REST_TREMOR}', or '{SPINE}'."
    )
