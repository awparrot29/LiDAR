"""What differs between a torso recording and a hand recording — and nothing else.

The two pipelines share the whole engine: orientation detection, depth sampling,
confidence and background rejection, depth smoothing, back-projection, CSV output
and rendering. The only things that actually differ are which model to run, which
landmarks to keep, which bones to draw, and which angles to compute. Those live
here as declarative tables so there is exactly one engine to maintain.

Feet are deliberately absent from the body profile. Measured on this project's
recordings, the heel and toe landmarks land only 5.5-7.6 px apart, giving a foot
length of 8-10 cm against a real ~25 cm — and LiDAR does not rescue it, because the
bottleneck is the 2D landmark spacing, not the depth. Adding them would produce
numbers that look real and are not.
"""
import mediapipe as mp

TORSO = "torso"
HAND = "hand"

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


def get(kind):
    """Profile for a subject kind: TORSO or HAND."""
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
                "max_depth_extent_m": 0.25}
    raise ValueError(f"Unknown kind '{kind}'. Use '{TORSO}' or '{HAND}'.")
