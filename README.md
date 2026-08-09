# LiDAR Gait Analysis

This project analyzes Parkinsonian movement using RGB-D video from LiDAR-enabled
iPhones/iPads, combining 3D pose estimation with depth data. It handles two kinds
of recording: a **torso** recording of someone walking, and a close-up **hand**
recording of finger movement. See [Torso vs Hand](#torso-vs-hand).

---

# Quick Start &mdash; process a recording

**`process_session.py` is the program to run.** Give it a zipped Stray Scanner
session; it writes the joint coordinate spreadsheets and a 3D stick figure movie.

```bash
pip install -r requirements.txt

# try it immediately on the recording included in this repository
python process_session.py example/sample_session.zip

# or on your own
python process_session.py my_session.zip
```

That is the whole workflow. Results land in a `<name>_results/` folder beside
the zip:

| Output | What it is |
|---|---|
| 18 CSV files | X&nbsp;Y&nbsp;Z per frame for twelve joints, plus knee, hip and elbow angles |
| `..._gait_skeleton_3d.mp4` | the tracked skeleton animated in 3D, three-quarter and side view |

Progress is shown while it runs:

```
  [###########################-]  97.5%  rendering movie 430/436      ~0m05s left
  Session recorded in landscape; frame 256x192, rotate=False
  Background model built from 200 frames; camera drift 0.2 cm
  Rejected 233 readings that were the background (4.5% of samples)
```

### Options

| Flag | Purpose |
|---|---|
| `--tracker mediapipe` | pose model (default). Faster. |
| `--tracker rtmpose` | RTMPose-m via ONNX Runtime. Different model, useful as a cross-check. |
| `-o FOLDER` | where to write results |
| `--no-movie` | spreadsheets only, skip the render |
| `--keep-graphs` | also save the per-joint PNG graphs |
| `--timeout N` | give up after N seconds (default 3600) |

```bash
python process_session.py my_session.zip -o results --tracker rtmpose
```

A 700-frame recording takes roughly four minutes on a laptop. Nothing needs to
be copied into the repository first, and no folders are left behind &mdash; the zip
is unpacked to a temporary directory that is removed afterwards.

### Getting the zip off the iPad

Record with **Stray Scanner** at **60 FPS**. In the **Files** app, find the
session folder, long-press it and choose **Compress**. That `.zip` is the input.

No device to hand? [`example/sample_session.zip`](example/) is a complete
recording ready to run &mdash; a subject rising from a chair and walking toward
the camera.

### Sharing a recording

Recordings identify whoever was filmed. Blur the face before passing one on:

```bash
python anonymize_session.py my_session.zip -o shareable.zip
```

The head is tracked through the whole recording and obscured in every frame,
while the depth and confidence maps pass through untouched. Measured on the
example, this shifts the tracked joints by 6&ndash;15&nbsp;mm, inside the
measurement noise, so an anonymised recording still analyses the same.

---

## Contents
- [Quick Start](#quick-start--process-a-recording)
- [Overview](#overview)
- [Results](#results)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Torso vs Hand](#torso-vs-hand)
- [Web App](#web-app)
- [Device Orientation](#device-orientation)
- [Denoising](#denoising)
- [Pose Trackers](#pose-trackers)
- [Interpreting the Output](#interpreting-the-output)
- [General-LiDAR](#general-lidar)
- [Gait-Analysis](#gait-analysis)

## Overview

This project aims to utilize iPad/iPhone LiDAR recording capabilities to analyze gait through 3D position tracking, specifically aimed to track and detect patterns in Parkinsonian gait. The repository has two main folders, general-lidar and gait-analysis. 

The general-lidar folder contains 5 programs designed for general-purpose LiDAR analysis applications, with functionalities such as creating depth maps of videos and/or individual frames, as well as 3D point cloud video/image recreations. You can read more about these programs in the [General-LiDAR](#general-lidar) section below.

The gait-analysis folder includes 4 programs for pose tracking and gait analysis. It utilizes the MediaPipe pose estimation model to map joint locations to the LiDAR data, which allows for data such as joint location over time and joint angles to be calculated and plotted. You can read more about the specific functionalities in the [Gait-Analysis](#gait-analysis) section below.

## Results

Participants were asked to stand up from a chair around five meters away from the camera and walk towards it. From this data, depth maps, 3D point cloud videos, and charts of joint angles and joint positions were generated. See below for some of the results.

Using LiDAR data, depth maps of euclidean distance from the camera at every pixel are generated from single frames of a video. By combining the depth maps from each frame of the video, a full LiDAR depth video is created:

<img src=sample-results/frame.png height="300"> <img src=sample-results/depthvid.gif height="290">

By using Open3D to visualize the 3D point clouds, interactive 3D frame-by-frame videos are created:

<img src=sample-results/3dvid.gif width="500">

For gait analysis, MediaPipe is used to track joint locations which can be mapped to the LiDAR depth data. Here is a visualization of MediaPipe pose tracking:

<img src=sample-results/mediapipe.gif width="300">

Using the MediaPipe pose tracking along with LiDAR data, graphs are generated like the ones below (similar graphs can be created for different human joints). The first graph displays the right knee angle over time which documents the transition from sitting to walking (fluctuations at the end reflect the knee leaving frame). The second graph shows the movement of the right knee's position over time compared to a trendline, and creates a detrended line which provides insights into how the knee is moving over time:

<img src=sample-results/rightkneeangle.png width="350"> <img src=sample-results/Rightkneedetrended.png width="350">

## Requirements
- Python 3.10 or newer
- iPhone 12 Pro+ or iPad Pro with LiDAR
- [Stray Scanner](https://apps.apple.com/us/app/stray-scanner/id1557051662) app to record LiDAR data
- [Anaconda](https://www.anaconda.com/) only if you want the Open3D point-cloud tools in `general-lidar/`

## Installation

```bash
git clone https://github.com/awparrot29/LiDAR.git
cd LiDAR
pip install -r requirements.txt
```

That covers [`process_session.py`](#quick-start--process-a-recording), the web
app and everything in `gait-analysis/`.

`env.yaml` describes the original Conda environment and is kept for the
`general-lidar/` scripts, which need Open3D. **It predates the newer work and
does not include `rtmlib`, `onnxruntime` or `imageio-ffmpeg`**, so the RTMPose
tracker and the H.264 movie will not work from it alone. Use `requirements.txt`
for anything in this README beyond the General-LiDAR section:

   ```bash
   conda env create -f env.yaml
   conda activate lidar-gait-analysis
   pip install -r requirements.txt
   ```

## Usage

> This section describes running the individual scripts directly, which is how
> the project began and is still how the `general-lidar/` tools work. For gait
> analysis you no longer need any of it &mdash;
> [`process_session.py`](#quick-start--process-a-recording) takes the zip
> straight from the iPad and does the whole job.

To utilize the LiDAR capabilities of the iPhone/iPad, you must have a LiDAR-equipped device (iPhone 12+ Pro/Pro Max, or iPad Pro).

The **Stray Scanner** app (available on the App Store) should be used to record LiDAR videos. When recording for gait analysis, press the 'Record new session' button and ensure the video is recorded at **60 FPS** for accurate joint tracking and analysis.

Each video recorded with Stray Scanner is saved to a folder in the **Files** app. This folder includes:
- Camera intrinsic files (camera_matrix.csv, imu.csv, odometry.csv)
- The original RGB video (rgb.mp4)
- Folders for depth and confidence frames (depth/, confidence/) as PNGs

Locate this folder in the Files app, rename it to your desired name, and move the folder into the **root directory** of your cloned LiDAR-Gait-Analysis folder. Once placed there, the programs are able process and analyze the video.

To run a [general-lidar](#general-lidar) program (ex. makevideo.py), ensure you are in the root directory of the cloned repo and run

   ```bash
   python3.10 general-lidar/makevideo.py
   ```

To run a [gait-analysis](#gait-analysis) program (ex. detrend.py), ensure you are in the root directory of the cloned repo and run

   ```bash
   python3.10 gait-analysis/detrend.py
   ```

## Torso vs Hand

Two kinds of recording are supported, and the pipeline handles both.

| | Torso | Hand |
|---|---|---|
| what it records | someone walking | a close-up of one hand |
| pose model | MediaPipe Pose | MediaPipe Hands |
| landmarks | 12 (shoulders, elbows, wrists, hips, knees, ankles) | 21 (wrist plus four joints per finger) |
| angles | 6 &mdash; elbows, hips, knees | 15 &mdash; three flexion angles per finger |
| tracker choice | MediaPipe or RTMPose | MediaPipe only |
| typical distance | 1.5&ndash;4 m | 0.2&ndash;0.5 m |

### Choosing

Both the web app and the standalone script **detect the subject automatically**,
and either lets you override it.

Detection uses lower-body visibility, which separates the two cleanly &mdash; measured
across every recording in this project:

| recording | nearest object | lower-body visibility | detected as |
|---|---|---|---|
| hand close-up | 203 mm | 0.00 | hand |
| park_sim | 1517 mm | 0.99 | torso |
| Jul_30 | 1360 mm | 0.99 | torso |
| Test3 (iPhone) | 1850 mm | 0.99 | torso |

There is no overlap, so the choice is not delicate. The evidence is logged on
every run, so a wrong call is visible rather than silent. Note that *hand
detection rate* is a poor signal on its own &mdash; MediaPipe finds a hand on the
walking subject 36&ndash;55% of the time &mdash; so it is only used as a tie-breaker.

On the **web app**, pick Auto-detect, Torso or Hand above the tracker choice. The
tracker row greys out for a hand, because RTMPose is a torso-only model.

With the **standalone script**, pass `--kind`:

```bash
python motion-analysis/process_session.py session.zip            # auto-detect
python motion-analysis/process_session.py session.zip --kind hand
python motion-analysis/process_session.py session.zip --kind torso
```

### Two differences that are not preferences

**Background rejection is on for torso, off for hand.** The background model learns
the persistent depth at each pixel, which identifies the static scene only when the
subject moves *through* it. A hand held in a close-up stays put, so it becomes the
persistent content and the model learns the hand itself as background. Measured on
a good hand recording, leaving it on rejected 40.9% of samples &mdash; including all
449 wrist readings &mdash; and cut usable depth from 79.4% to 38.4%.

**A depth-extent limit replaces it for hands.** A hand is only about 20 cm across,
so no joint can sit 25 cm in depth away from the rest of it; a fingertip that
samples the wall behind it lands outside that extent and is discarded. Without
this, a single bad fingertip was thrown metres away, dragging one bone across the
whole stick figure. The torso limit is a permissive 1.5 m, because a walking
person's limbs genuinely span depth.

### Checking a hand recording is usable

Watch the **usable depth** percentage. Two of three hand recordings taken for this
project were silently defective: their ARKit confidence channel was zero
everywhere and the depth read 2.4&ndash;3.7x too far, while looking perfect to the eye
&mdash; the crisp outline in a depth map comes from the RGB image, not the sensor. The
confidence gate catches this, rejecting every sample from both. A near-zero figure
means re-record. Healthy figures are around 79% for both subjects.

### Known limitation: feet

Feet are deliberately excluded. The heel and toe landmarks land only 5.5&ndash;7.6 px
apart, giving a foot length of 8&ndash;10 cm against a real ~25 cm, and LiDAR does not
rescue it because the bottleneck is the 2D landmark spacing rather than the depth.
Including them would produce numbers that look real and are not.

---

## Web App

The upload page now has a **Subject** choice above the tracker: Auto-detect,
Torso or Hand. See [Torso vs Hand](#torso-vs-hand). The tracker row greys out
for a hand recording, because RTMPose is a torso-only model.

`app.py` wraps the gait-analysis pipeline in a Flask web app so a session can be
processed without a local Python environment. Upload a zipped Stray Scanner
folder, pick a tracker, and it returns a ZIP of the joint coordinate and angle
CSVs. A progress bar reports the frame being tracked and an estimated time
remaining.

Run it locally with:

   ```bash
   pip install -r requirements.txt
   python app.py
   ```

then open http://localhost:5000.

The app extracts the upload to a temporary directory, runs `calculateangle.py`
in a subprocess there, and bundles `charts/<session>/data/*.csv` into the
response. Nothing is written to the repository and uploads are deleted once the
job finishes.

### What you get back

- **A 3D stick figure movie**, played in the page as soon as it is ready. Two
  panels: a three-quarter view and a side view turned 90&deg; about the vertical
  axis, where travel in depth reads as left-to-right motion. Every axis is a
  true distance from the camera, so the subject moves through the scene.
- **The coordinate CSVs** — one per joint with X&nbsp;Y&nbsp;Z per frame, plus
  angle CSVs for knees, hips and elbows.

The movie is downloadable on its own, and is also included in the results ZIP.
It is encoded H.264 through `imageio-ffmpeg`'s bundled libx264 rather than
OpenCV's `mp4v`, because no browser will play MPEG-4 Part 2 in a `<video>`
element. That also cuts a typical clip from ~2.9&nbsp;MB to ~0.24&nbsp;MB.

### Deploying to Azure App Service

The app runs on a Free (F1) Linux Python 3.11 plan. Two deployment details are
easy to get wrong:

- **Set the startup command to `gunicorn --bind=0.0.0.0 --timeout 600 app:app`
  directly.** Wrapping it in a shell script fails with exit code 127, because a
  bash script does not inherit the Oryx virtual environment on its PATH.
- **`apt-get` during the build does not persist to the runtime container** —
  only `/home` and the built virtualenv carry over. Missing shared libraries
  must be resolved at the Python-package level instead. `postbuild.sh` does this
  for OpenCV: both MediaPipe and rtmlib depend on GUI OpenCV builds that need
  `libGL`/`libxcb`, so it swaps them for `opencv-contrib-python-headless` after
  the install step.

`GET /healthz` reports the import status of every native dependency, which is
the quickest way to verify a deployment.

## Device Orientation

The RGB video is always stored landscape (1920x1440) no matter how the device
was held. A session shot in **portrait** therefore has the subject lying
sideways in the raw frames and needs a 90&deg; rotation to stand them up; a
session shot in **landscape** is already upright and must not be rotated.
Rotating one anyway lays the skeleton on its side, so it appears to stand on a
wall.

`sessiongeom.py` decides which case applies from the gravity vector in
`imu.csv` — gravity sits on the device *y* axis in portrait and the *x* axis in
landscape. Units differ between Stray Scanner versions, so only the dominant
axis is used. A session with no usable `imu.csv` is assumed portrait, matching
how everything recorded before this check was handled.

The same module derives the camera intrinsics for the frame the tracker works
in. The two cannot be separated, because the rotation swaps the x and y axes of
the camera matrix. Both orientations place the principal point at the centre of
the frame, giving anatomically sensible measurements (shoulder width ~0.35 m,
thigh ~0.41 m).

> Coordinates exported before this was added are **not comparable** with newer
> ones. `calculateangle.py` stamps a schema number into each
> `landmark_cache.npz` and discards caches written by an older version, since
> some session ZIPs carry them.

## Denoising

Raw coordinates jerk about, because each joint's distance is a single number
pulled from a 192&times;256 depth map. Two stages clean this up, applied
automatically by `process_session.py` and the web app.

**Background rejection** (`background.py`). A thin limb against a distant wall
can fail to register, so the pixel under the landmark reports the scene behind
the subject. That is not noise and no amount of averaging repairs it. The camera
is stationary, so the scene is learned from the recording: for each pixel the
background is the 90th percentile of its depth history, and a reading within
50&nbsp;mm of it is marked untrusted.

> The percentile must be high rather than the median. A median model is
> contaminated wherever the subject lingers over a pixel &mdash; measured on
> `park_sim`, it placed four of five test joints *behind* their own
> "background". Rejected readings are marked untrusted rather than discarded;
> discarding them left gaps that the filter turned into metre-scale artefacts.

**Outlier-rejecting average** (`depthsmooth.py`). Each joint's depth is then
averaged over nine frames (150&nbsp;ms), dropping samples more than two standard
deviations from the window mean before averaging the rest, iteratively because
one extreme value inflates the deviation enough to hide itself on a single pass.

> The residual at the extremities is not steady noise but rare and violent:
> kurtosis 47 at the ankle against 3.0 for a normal distribution. A plain mean
> smears such a spike across the window instead of removing it. On a synthetic
> 583&nbsp;mm spike the clipped mean leaves 1.9&nbsp;mm of error where a plain
> nine-frame mean leaves 63&nbsp;mm.

The window is centred, so constant-velocity motion passes through untouched and
a 1&nbsp;Hz limb swing keeps 97% of its amplitude.

Measured on `park_sim`, walking phase:

| | before | after |
|---|---|---|
| jumps over 50&nbsp;mm in one frame | 153 | **17** |
| worst ankle jump | 666&nbsp;mm | **113&nbsp;mm** |
| ankle step per frame | 12.7&nbsp;mm | **7.2&nbsp;mm** |
| knee-angle range | 85.3&deg; | 86.2&deg; *(signal not flattened)* |

Every run prints what it rejected, so the cleaning is never silent.

**Not tried and rejected:** sampling a 5&times;5 patch around each joint instead
of one pixel. A forearm is only about five pixels wide at 2.4&nbsp;m, so the
patch is mostly background and its median jumps to the wall &mdash; it put the
wrist more than half a metre behind the torso in 5.7% of frames. Sampling stays
at the single pixel.

## Pose Trackers

Two interchangeable pose backends are available; both produce identical output
formats, so they can be swapped freely.

| Module | Model | Keypoints | Notes |
|---|---|---|---|
| pipelandmark.py | MediaPipe Pose (heavy) | 33-point BlazePose | Original tracker. Bundles its own runtime. |
| pipelandmark_rtmpose.py | RTMPose-m via rtmlib | COCO 17-point | Runs on ONNX Runtime — no PyTorch, mmcv or mmdet. |

`pipelandmark_rtmpose.py` is a drop-in replacement: swap the import in
`calculateangle.py`, or select the tracker in the web app. It uses rtmlib's
`PoseTracker` with `det_frequency=10`, reusing each person-detection box for ten
frames. Detection is the expensive half of the pipeline, and at 60 FPS the
subject barely moves in that window, so this runs roughly eight times faster
than detecting every frame while staying within half a pixel of it. Model
weights (~155 MB) download automatically to `~/.cache/rtmlib` on first use.

Measured on a 717-frame session, the two trackers agree to a median of about
2 cm per joint.

### Single-pass tracking

`extract_all_landmarks(folder)` tracks every landmark in **one** pass over the
video and returns `{landmark name: [(x, y, z), ...]}`. `calculateangle.py` calls
it once and slices out the joints it needs. Earlier versions called
`extract_landmarks()` once per joint, re-running the whole video six times;
avoiding that cut a 458-frame session from 424 s to 111 s. `extract_landmarks()`
is still available as a wrapper for the three-landmark call style.

## Interpreting the Output

**Limbs pointing at the camera are foreshortened.** One depth sample is taken
per joint, and that sample is the body's *front surface*, not the joint centre.
When a limb runs along the viewing axis its two ends return almost the same
depth, so the limb collapses.

This shows up clearly during the sit-to-stand at the start of a recording, where
the thighs point straight at the camera. Measured on `park_sim`:

| | seated / standing up | walking |
|---|---|---|
| torso lean, p95 | 24.6&deg; | 4.9&deg; |
| torso length | 0.39 m (0.33&ndash;0.54) | 0.53 m (IQR 0.02) |
| thigh length | 0.20 m | 0.34 m |

A thigh of 0.20 m is 40% short. The walking segment is stable and trustworthy;
the seated segment mixes genuine trunk flexion — you must lean forward to rise
from a chair — with unreliable geometry, and should not be used quantitatively.

A quick sanity check on any session: bone lengths should stay roughly constant
over time. Where they do not, the joints in question are unreliable for those
frames.

**Depth is the trustworthy axis.** `z` is read straight from the LiDAR map and
is accurate to 1&ndash;2%. X and Y are reconstructed through the camera model
and depend on the intrinsics being right for the session's orientation.

## Repository layout

| folder | what it is |
|---|---|
| `gait-analysis/` | the original torso pipeline (`pipelandmark.py`, `calculateangle.py`, `skeleton3d.py`) |
| `motion-analysis/` | **one program for both subjects** &mdash; torso or hand, auto-detected. Shares one engine; the only differences live in `profiles.py`. |
| `nolidar-gait-analysis/` | the same pipeline with the depth sensor removed, plus `compare.py` which scores it against the LiDAR version on physical invariants |
| `tremor-analysis/` | spectral quantification of shaking. Parked &mdash; built ahead of need, and no recording containing real tremor exists yet. |
| `general-lidar/` | standalone LiDAR utilities (depth maps, point clouds) |
| `example/` | one face-blurred sample session |

---

## General-LiDAR

The general-lidar folder contains 5 programs whose functionalities are described below. These programs are general-purpose LiDAR data analysis and visualization tools as opposed to specific tools targeted for gait analysis.

**Note: The video dimensions are 192x256, so x-coordinates range from 0-191 and y-coordinates range from 0-255**

### General-LiDAR Summary

| Script                | Description                                                         |
|------------------------|---------------------------------------------------------------------|
| png2realdepth.py     | Converts a single video frame to a depth map and outputs distance from the camera |
| distancefrompoints.py| Creates a depth map from a user-defined point; computes distance between any two pixels |
| makevideo.py         | Generates a LiDAR-based depth/confidence video and a side-by-side comparison with the RGB video |
| 3dframe.py           | Creates and saves a 3D point cloud of a single frame as a PLY file |
| 3dvid.py             | Creates and plays an interactive 3D point cloud video; saves it as MP4 |


**png2realdepth.py**: This program converts a frame from a video into a depth map of euclidean distances from the camera, and can display the distance value in meters from any point in the frame.

It first prompts the user to input their LiDAR data folder name and an output file name (which should be in PNG format). It then prints the number of total frames in the input video, and asks the user to pick a valid frame number (frames start from 000000 and should be exactly 6 digits). It then prompts the user to choose an x and y pixel coordinate at which the real distance from that point to the camera is printed, and a depth map of all points is also saved to the output file. It also opens a visualizer window showing the depth map, and depth values in the visualizer can be viewed by hovering the cursor over pixels. 

**distancefrompoints.py**: This program creates a depth map of distances from a specific point in the frame that the user inputs, also with the ability to print the distance between any two points in the frame.

It first prompts the user to input their LiDAR data folder name and an output file name (which should be in PNG format). It then prints the number of total frames in the input video, and asks the user to pick a valid frame number (frames start from 000000 and should be exactly 6 digits). It then prompts the user to choose two sets of x and y pixel coordinates. The first coordinate location input by the user will be the reference coordinate from which the distance map data will be based on (it will be a map of the distances from this point to all other points). The second coordinate location will be the point from which a distance will be printed, allowing the user to print distances between these two exact pixel locations. The distance map is then saved to the output file. It also opens a visualizer window showing the distance map, and each distance can be viewed by hovering the cursor over each pixel. 

**makevideo.py**: This program creates either a depth or confidence map recreation of the video based on LiDAR data. It also creates a video with the depth/confidence video side-by-side with the original.

It first prompts the user to input their LiDAR data folder name, then asks if they want a depth map or a confidence map. It then asks for two output file names, one for the standalone depth/confidence video, and one for the comparative video (these should both be in MP4 formats). It then outputs one standalone video in the first output file which shows the depth/confidence version of the original video, and also a comparative side-by-side video in the second output file of the depth/confidence video next to the original RGB video.

**3dframe.py**: This program opens an Open3D window with and interactive 3D point cloud model of a single frame of a video, and saves it as a PLY file.

It first prompts the user to input their LiDAR data folder name and an output file name (which should be in PLY file format). It then prints the number of total frames in the input video, and asks the user to pick a valid frame number (frames start from 000000 and should be exactly 6 digits). It then creates a 3D point cloud representation of the LiDAR data, shows the point cloud in an open3D window, and saves the result to the output file.

**3dvid.py**: This program opens an Open3D window where an interactive 3D point cloud recreation of the video plays. The result of this video is saved to an MP4 file.

It first prompts the user to input their LiDAR data folder name, as well as an output file name (which should be in .mp4 format) then opens an open3D visualizer window in which the 3D point cloud video plays. The result of this 3D visualizer is saved to the output file.

## Gait-Analysis

The gait-analysis folder contains 4 programs whose functionalities are described below. These programs utilize the MediaPipe pose estimation model to track joint locations over time, and map the joints to the LiDAR depth data which allows for analysis of gait through 3D pose tracking.

### Gait-Analysis Summary

| Script              | Description                                                                 |
|---------------------|-----------------------------------------------------------------------------|
| visualize.py      | Overlays MediaPipe pose landmarks on the video and saves it as an MP4       |
| calculateangle.py | Calculates joint locations and angles over time; saves CSV + graphs           |
| detrend.py        | Plots joint movement over time, removes trendline for better insight        |
| pipelandmark.py   | Helper module for joint location extraction (not used on its own)         |
| pipelandmark_rtmpose.py | Same helper backed by RTMPose instead of MediaPipe (not used on its own) |
| sessiongeom.py    | Works out portrait vs landscape from imu.csv and derives the matching intrinsics |
| background.py     | Learns the static scene so readings that are the wall or floor can be rejected |
| depthsmooth.py    | Averages each joint's depth across frames, dropping outliers beyond two sigma |
| skeleton3d.py     | Renders the 3D stick figure movie from the coordinate CSVs                 |

**visualize.py**: This program displays the MediaPipe joint landmarks drawn onto the video, and saves it as an MP4.

It first prompts the user to input their LiDAR data folder name as well as an output file name (which should be in MP4 format). It then plays the original RGB video with pose estimation joint landmarks from MediaPipe drawn onto the video, saving this to the output file.

**calculateangle.py**: This program computes the elbow, knee, and hip angles over time. It saves this data as well as the 3D location data for each body part (joints) over time as CSV files, and also generates graphs of the data over time. These are saved to the charts folder in "data" and "graphs" subfolders, respectively.

It first prompts the user to input their LiDAR data folder name. In the charts/data folder (which it creates if it doesn't already exist), it creates CSV files for the 3D location of each body part over time, as well as CSV files containing the elbow, knee, and hip angle data in degrees over time. It also creates graphs in PNG format to visualize the distance of each of the body parts from the camera over time and graphs for the angle over time. These are in charts/graphs.

**detrend.py**: This program runs calculate angle if it hasn't already been run, and in addition creates a PNG graph of the distance of body parts (joints) from the camera over time plotted alongside a trendline, and a detrended line is generated.

It first prompts the user to input their LiDAR data folder name. It then creates a PNG graphs of the distance of each body part from the camera over time along with a trendline. It also plots a detrended line by plotting the difference between the actual distance and the trendline over time. These PNG's are saved to charts/detrended/

**pipelandmark**: This program is a helper module and not meant to be executed directly. It is automatically imported by calculateangle.py and detrend.py which require joint landmark extraction.
