# Example recording

`sample_session.zip` is a complete Stray Scanner session you can run straight
away, so the pipeline can be tried without a LiDAR device to hand.

```bash
python process_session.py example/sample_session.zip
```

Results appear in `example/sample_session_results/`: eighteen CSV files and the
3D stick figure movie. A run takes roughly four minutes on a laptop.

## What is in it

The `park_sim` recording, the one behind the figures in the main README. A
subject stands up from a chair about four metres away and walks toward the
camera, with a simulated Parkinsonian gait.

| | |
|---|---|
| Frames | 716 at 60 FPS, about 12 seconds |
| Video | `rgb.mp4`, 1920&times;1440 |
| Depth | 717 PNG frames, 256&times;192, millimetres |
| Confidence | 717 PNG frames |
| Also | `camera_matrix.csv`, `imu.csv`, `odometry.csv` |

Recorded in portrait, so it also exercises the rotation handling described under
[Device Orientation](../README.md#device-orientation).

## The subject's face is blurred

This is a research recording of a person, so the face is obscured in every one
of the 716 frames before publication, using
[`anonymize_session.py`](../anonymize_session.py):

```bash
python anonymize_session.py my_session.zip -o shareable.zip
```

The head is located from MediaPipe's pose landmarks rather than a face detector,
which loses a subject several metres away. Depth and confidence maps are
192&times;256 silhouettes with no facial detail and are copied through untouched.

Blurring moves the tracked joints by a mean of 6&ndash;15&nbsp;mm, within the
measurement noise. Against the unblurred original the gait figures are
effectively unchanged: knee-angle range 86.2&deg; against 84.1&deg;, walk
distance 2.93&nbsp;m against 2.92&nbsp;m, and the same count of frame-to-frame
jumps over 50&nbsp;mm. The recording is still a fair demonstration of what the
pipeline produces.

**Use `anonymize_session.py` before sharing any recording of a participant.**
