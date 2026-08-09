# NoLiDAR-Gait-Analysis

The gait pipeline with the depth sensor removed. Produces the same joint
coordinate and angle CSVs from **RGB video alone**, so it runs on any iPhone or
iPad — no LiDAR, no Pro model.

A clone of `../LiDAR-Gait-Analysis/`, not a replacement for it. Both are kept so
they can be compared, and they share one copy of the recordings in
`../Examples/`.

## Why this exists

The clinical trial is already recording ordinary video of patients getting up and
walking, and the clinic has declined to add LiDAR capture. If the analysis needs
nothing but the video, that obstacle disappears — and existing footage may be
analysable as-is.

## What changed from the LiDAR version

**Only the source of z.** The 2D landmarks and the back-projection

```
x = (col - cx) * z / fx        y = (row - cy) * z / fy
```

are the same operation as `pipelandmark.py`. That is deliberate: it makes the two
pipelines a controlled comparison rather than two different programs, so any
difference in the output is attributable to depth alone.

Depth now comes from MediaPipe's metric body model in two steps:

1. `pose_world_landmarks` places every joint in metres relative to the hip
   midpoint — the body's *shape*, including relative depth. This is a learned
   anatomical model, not a measurement.
2. That is hip-relative, so it is anchored to an absolute distance recovered from
   apparent torso size by similar triangles, `z0 = physical_span * f / pixel_span`,
   median over six torso spans. Measured against LiDAR on `park_sim`: **0.99×,
   correlation 0.984** across the walk.

`depth/` and `confidence/` are never read. `background.py` and `depthsmooth.py`
are gone — there is no depth map to reject background from, and **no temporal
smoothing is applied at all**. Whether one is needed is measured by `compare.py`
rather than assumed.

## Files

| file | purpose |
|---|---|
| `poselandmark.py` | the tracker — drop-in replacement for `pipelandmark.py` |
| `run_examples.py` | regenerate clean output for every example, both pipelines |
| `compare.py` | score LiDAR vs no-LiDAR against physical invariants |
| `sessiongeom.py` | copied unchanged; orientation from `imu.csv`, per-session intrinsics |
| `calculateangle.py` | copied for reference; `run_examples.py` computes angles itself |

## Usage

```
# regenerate all example output (both pipelines)
python run_examples.py

# just this solution, one session
python run_examples.py --only nolidar ../Examples/torso/park_sim

# score the two against each other
python compare.py
```

Works on a session with no `depth/` folder at all — `poselandmark.py` falls back
to deriving the working frame from `rgb.mp4`, `camera_matrix.csv` and `imu.csv`.
When `depth/` *is* present it matches the LiDAR pipeline's working resolution, so
both see identical input frames and the comparison stays fair.

## How "better" is defined

Neither pipeline has ground truth, so `compare.py` does not try to measure error
directly. It tests each reconstruction against facts that hold in the real world
regardless of what any sensor says:

- **A. Rigid-segment constancy** — a femur does not change length while you walk.
  The primary test, and the strongest evidence available, because it is a hard
  physical constraint needing no external reference.
- **B. Bilateral symmetry** — left and right bones are near-equal in length, true
  even in pathological gait.
- **C. Absolute plausibility** — constancy alone is not enough; a method could be
  rock-steady and wrong. A and C together separate precision from accuracy.
- **D. Floor planarity** — stance-phase ankles must lie on one plane.
- **E. Path straightness and scale** — context; the walk is only roughly straight,
  so this is weaker evidence.
- **F. Signal quality** — jitter and impossible velocities, reported last and
  labelled, because **smoothness is not accuracy**: a constant output has zero
  jitter and infinite error.

## Known limitations

- Depth is a **model estimate**, so it carries the model's priors. Expect it to be
  weakest where the body is foreshortened along the view axis.
- **Not validated on pathological gait.** The model was trained mostly on healthy
  subjects and could regularize impaired movement toward normal — the very signal
  being measured. Cross-check against 2D-derived metrics before trusting clinical
  numbers.
- Absolute segment lengths are **not yet anchored to the subject**. Section C of
  the comparison stays inconclusive until someone tape-measures the subject's
  thigh and shin, the way the hand was measured at 18.0 cm.
- Feet are unreliable: MediaPipe collapses the foot in 3D (measured 7.6 cm against
  a real ~25 cm), so toe tapping and leg agility need a different approach.
