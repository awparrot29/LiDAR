# Motion-Analysis

One program for both subjects. Point it at a Stray Scanner recording and it
extracts joint coordinates, computes joint angles, writes CSVs, and draws a
two-view 3D stick figure so the output can be verified by eye.

```
python process_session.py <folder or .zip>          # detects torso or hand
python process_session.py session.zip --kind hand   # force hand
python process_session.py session.zip --kind torso  # force torso
python process_session.py session.zip --no-movie --no-graphs -o results
```

Output, under `<session>/output/<kind>/`:

| file | contents |
|---|---|
| `data/<landmark>.csv` | x, y, z in metres, one row per frame |
| `data/<joint> angle.csv` | degrees, one row per frame |
| `graphs/*.png` | angle and z-distance traces |
| `skeleton.mp4` | two-view 3D stick figure |
| `summary.txt` | tracking rate and per-angle ranges |

Torso output matches `LiDAR-Gait-Analysis` exactly — same 12 landmarks, same six
angles, same file layout — so anything downstream works unchanged. Verified on
`Jul_30`: this program and the original pipeline reject the same 422 background
readings (12.6%, most affected left shoulder with 62).

## Why one program instead of two

A torso run and a hand run share the whole engine: orientation from `imu.csv`,
depth sampling, confidence and background rejection, depth smoothing,
back-projection, CSV writing and rendering. The only real differences are which
MediaPipe model runs, which landmarks to keep, which bones to draw, and which
angles to compute — four declarative tables in `profiles.py`.

Keeping them as two programs would mean two copies of the engine. This project has
already been bitten once by duplicated landmark tables drifting apart, and Phase 5
of the brief asks for one integrated package.

| | torso | hand |
|---|---|---|
| model | MediaPipe Pose | MediaPipe Hands |
| landmarks | 12 | 21 |
| angles | 6 (elbows, hips, knees) | 15 (3 per finger) |

## Subject detection

Auto-detected, with `--kind` to override. Measured on all four recordings:

| recording | nearest object | lower-body visibility | hand found |
|---|---|---|---|
| hand close-up | 203 mm | **0.00** | 100% |
| park_sim | 1517 mm | **0.99** | 45% |
| Jul_30 | 1360 mm | **0.99** | 36% |
| Test3 (iPhone) | 1850 mm | **0.99** | 55% |

Lower-body visibility separates them completely — 0.00 against 0.99, nothing in
between — with nearest-object distance corroborating. The evidence is printed on
every run, so a wrong call is visible rather than silent.

Note that hand-detection rate is a **poor** signal alone: MediaPipe finds a hand on
the walking subject 36–55% of the time. It is used only as a tie-breaker.

## Two things that differ by subject for real reasons

**Background rejection is ON for torso, OFF for hand.** Not a tuning preference.
`background.build_model` learns the persistent depth at each pixel, which
identifies the static scene only when the subject moves *through* it. A hand held
in a close-up stays put, so it becomes the persistent content and the model learns
the hand itself as background. Measured: that rejected 40.9% of samples including
**all 449 wrist readings**, cutting usable depth from 79.4% to 38.4%.

**Depth-extent limit replaces it for hands.** A hand is only ~20 cm across, so no
joint can sit 25 cm in depth from the rest of it. A fingertip that samples the wall
behind it lands far outside that extent and is caught. Without this, a single bad
fingertip was flung metres away, dragging one bone across the whole stick figure
and shrinking the rest of the skeleton to a dot. Torso uses a permissive 1.5 m,
because a walking person's limbs genuinely span depth.

## Checking a recording is usable

Watch the **usable depth** percentage. Two of three hand recordings taken for this
project were silently defective: their confidence channel was zero everywhere and
depth read 2.4–3.7× too far, while looking perfect to the eye because the crisp
outline in a depth map comes from the RGB image. The confidence gate catches this —
it rejected every sample from both. A near-zero figure means re-record.

Healthy figures on the good recordings: hand 79.4%, torso 78.7%.

## Known limitation: feet

Feet are deliberately **not** in the torso profile. The heel and toe landmarks land
only 5.5–7.6 px apart, giving a foot length of 8–10 cm against a real ~25 cm — and
LiDAR does not rescue it, because the bottleneck is the 2D landmark spacing, not the
depth. Including them would produce numbers that look real and are not. Toe tapping
and leg agility need a different approach.

## Files

| file | purpose |
|---|---|
| `process_session.py` | CLI — session in, everything out |
| `profiles.py` | the only things that differ between torso and hand |
| `detect.py` | torso-or-hand detection, with the evidence |
| `extract.py` | the shared engine: depth sampling → 3D coordinates |
| `angles.py` | joint angles, CSVs, graphs, summary |
| `skeleton3d.py` | two-view stick figure video |
| `sessiongeom.py`, `background.py`, `depthsmooth.py` | copied unchanged from the gait pipeline |

## Related folders

- `../LiDAR-Gait-Analysis/` — the original torso pipeline and the web app
- `../NoLiDAR-Gait-Analysis/` — the same pipeline with no depth sensor, plus the
  head-to-head comparison against LiDAR
- `../Tremor-Analysis/` — spectral shake quantification. **Parked** — built ahead of
  need; revisit once joint extraction is settled and a recording with actual
  shaking exists.
- `../Examples/` — one copy of every good recording, torso and hand
