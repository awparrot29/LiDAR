# Tremor-Analysis

Maps the 3D location of finger and body joints over time, and quantifies shaking
from it. **No LiDAR** — runs on any iPhone or iPad.

Named for the goal rather than the body part, because the same engine takes either
the 21 hand landmarks or 16 body joints. Rest tremor in MDS-UPDRS covers arms and
legs as well as hands, so the analysis is shared.

## Quick start

```
python process_session.py <session folder or .zip>
python process_session.py hand_session --reference wrist     # isolate finger tremor
python process_session.py session.zip --kind body            # arms and legs
```

Output lands in `<session>/output/tremor/`:

| file | contents |
|---|---|
| `data/<joint>.csv` | per joint per frame: `x_px, y_px, x_mm, y_mm, z_mm` |
| `tremor.csv` | per joint per axis: amplitude, frequency, SNR, constancy |
| `tremor_report.txt` | the same, readable, with the self-checks |
| `landmarks.npz` | raw arrays, so re-analysis needs no re-tracking |
| `skeleton.mp4` | two-view 3D skeleton |

## How shaking is measured

**Frequency, not counting.** Tremor is an oscillation, so amplitude and rate come
from the spectrum. That deliberately avoids counting discrete movements, which is
fragile exactly when amplitude decrements — the case that matters clinically.

Three bands:

| band | range | what lives there |
|---|---|---|
| voluntary | 0.2–2 Hz | reaching, opening a fist, walking |
| **tremor** | **3–8 Hz** | Parkinsonian rest tremor sits at 4–6 Hz |
| noise | 12–25 Hz | above physiological tremor, below Nyquist at 60 fps |

The noise band is the useful trick: it gives a **per-joint, per-recording noise
estimate** with no separate baseline recording needed, so every amplitude is
reported alongside the SNR that earned it. A joint counts as tremulous only at
SNR ≥ 3 against its own noise.

Reported per joint: peak-to-peak amplitude, dominant frequency, peak sharpness
(how concentrated the oscillation is — separates a real tremor from broadband
noise with a bump in it), and **constancy**, the fraction of 2 s windows in which
tremor is present.

### The voluntary-movement guard — why it exists

Clearing the noise floor is **not sufficient**. Voluntary movement is not a pure
sinusoid, so a vigorous 1–2 Hz task throws harmonics straight into the 3–8 Hz
tremor band and passes an SNR-against-noise test easily.

This was not hypothetical. The first run of this program on a healthy hand doing
slow open/close reported **tremor on 19 of 21 joints** at 3–4 Hz with SNR up to
15. The giveaway was that amplitude grew with distance from the wrist — 3–5 mm at
the knuckles, 17–27 mm at the fingertips. That is finger flexion, not tremor,
which would be far more uniform across the hand.

So a joint is only called tremulous when it clears the noise floor **and** its
tremor-band amplitude is not swamped by the voluntary band
(`voluntary_ratio <= 3`). With the guard, 18 of those 21 joints are correctly
reported as `masked by movement` instead. The `vol/trem` column is printed so a
borderline case can be judged rather than hidden.

The deeper lesson is a protocol one: **assess tremor on a recording of the limb at
rest**, not during a movement task. Rest tremor is defined at rest for exactly
this reason.

## The measurement that shaped the design

Noise floor inside the tremor band, measured on palm landmarks during a
non-tremulous recording (they stay nearly still while the fingers move, so their
3–8 Hz content is essentially pure noise):

| axis | noise floor |
|---|---|
| **2D image plane → mm** | **1.05–1.22 mm** |
| 3D in-plane | 1.28–5.61 mm |
| **3D depth** | **10.67–12.05 mm** |

**Depth is ~10× noisier than the image plane**, so it cannot resolve a 1 cm
tremor. Two consequences, both built in:

1. **In-plane and depth are reported separately and never combined.** Averaging
   them would let the noisy axis dominate.
2. **Point the camera so the shaking crosses the frame, not toward the lens.**
   The same geometry lesson as recording a transverse walk.

For scale: a 1 cm tremor sits ~9× above the in-plane floor, so clinically
relevant amplitudes are comfortably measurable — in-plane.

## Why there is no depth sensor here

On this project's recordings, LiDAR depth was ~10× noisier than the 2D track
inside the tremor band, and for a close-range hand it was wrong by 2.4–3.7× on two
of three recordings — silently, with a depth map that looked perfect because the
crisp outline comes from the RGB image. See `project_lidar_iphone_vs_ipad` in
memory. Absolute 3D here comes from `cv2.solvePnP` on the model's metric landmarks
instead, which is geometry rather than a learned depth estimate.

## Per-subject scale

The model knows how big a generic hand is, which is what breaks scale ambiguity.
For a specific subject, anchor it with a ruler and pass `--scale`.

Measure **wrist to middle fingertip with the hand extended** and compare against
the value the report prints for the most-extended frames.

For the current subject that measurement is **18.0 cm**, and the model reads
18.00 cm — so `--scale 1.0` is correct and no correction is needed.

**A trap worth knowing:** wrist-to-fingertip is *not* rigid — it shortens from
~18 cm open to ~8 cm closed. Comparing a ruler value against the *median* over an
open/close recording is meaningless. Use the most-extended frames only. The
`rigid spans` block in the report exists for this reason: those spans (wrist to
middle knuckle, knuckle to knuckle) genuinely cannot change, so their variability
is a fair check on the 3D.

## Checks the report runs on itself

- **Camera motion** from `odometry.csv`. Camera shake is indistinguishable from
  real tremor in the absolute measurement, so a moving device invalidates it. Under
  1 cm of displacement passes; more raises a warning.
- **Rigid spans** — lengths that cannot change. If they wander more than ~10%, the
  3D is not trustworthy and the report says so.
- **Tracking rate** — fraction of frames with a detected subject.

## `--reference`

Subtracting a joint first changes what is being measured:

- `--reference wrist` isolates **finger** tremor from whole-hand movement.
- No reference keeps the joint's absolute motion, which is what rest tremor of a
  limb means — but it is also the case where camera shake contaminates the result,
  so check the camera-motion line.

## Not yet validated

**No recording containing actual shaking exists yet.** Everything so far is a
healthy hand doing slow open/close, so the noise floor and the geometry are
measured but tremor *detection* is not.

The open question, and it is a real risk: **does MediaPipe low-pass the tremor
away?** A pose model may treat a 5 Hz oscillation as jitter and smooth it out —
attenuating precisely the signal wanted. This is the same failure mode found in the
LiDAR pipeline's denoiser, which was suppressing real joint velocity about
two-fold.

The cheap test: shake a hand in time with a **metronome at a known rate** (say
5 Hz), held so the motion crosses the frame, for 20–30 s. If the report comes back
with 5 Hz at the right amplitude, the model preserves tremor. If the amplitude is
attenuated, a different tracker is needed.

A longer recording also helps. At 7.5 s the frequency resolution is 0.13 Hz and
constancy has few windows to work with; 20–30 s is better on both counts.

## Files

| file | purpose |
|---|---|
| `landmarks.py` | extraction — hand or body, 2D track plus absolute 3D via solvePnP |
| `tremor.py` | the spectral engine: bands, SNR, dominant frequency, constancy |
| `process_session.py` | CLI: session in, CSVs + report + video out |
| `render.py` | two-view skeleton video |
| `sessiongeom.py` | copied from the gait pipeline; orientation from `imu.csv` |
