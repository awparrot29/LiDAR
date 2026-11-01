# Offline command-line program (`lidar_cli.py`)

`lidar_cli.py` does what the web app does on a computer with **no internet
access**. Give it a Stray Scanner recording and the MDS-UPDRS Part 3 test it
was recorded for. It writes a ZIP holding:

- one CSV per joint position (`time_s, x_m, y_m, z_m`) and per joint angle
  (`time_s, angle_deg`), the same files the website returns
- the 3D skeleton, camera, and LiDAR videos
- `README.txt`, which explains every file in that ZIP: what each joint and
  angle is, the units, the coordinate system, how much of each recording had
  usable data, and which files matter for the chosen test

## Setup

The analysis runs fully offline, but you need internet access **once**, on any
computer, to collect two things: the Python packages and the model weights.

### 1. On a computer with internet access

Use the same operating system and the same Python version (3.11, 64-bit) as the
offline computer, so the downloaded packages match it.

```bash
git clone <this repository>
cd LiDAR-Gait-Analysis

# Python packages, saved as files
pip download -r requirements-cli.txt -d wheels

# Model weights (~0.6 GB) into models/
pip install -r requirements-cli.txt
python lidar_cli.py --download-models
```

Copy the whole folder, including `wheels/` and `models/`, to the offline
computer on a USB drive or a network share. `models/` is not in git because
several of its files are over GitHub's 100 MB limit.

### 2. On the offline computer

```bash
cd LiDAR-Gait-Analysis
python -m venv .venv
.venv\Scripts\activate           # Windows
# source .venv/bin/activate      # macOS / Linux
pip install --no-index --find-links wheels -r requirements-cli.txt

python lidar_cli.py example/sample_session.zip --test 3.10   # check it works
```

MediaPipe's models are built into its package. If you only need the MediaPipe
tests (hand tests 3.4-3.6 and 3.15-3.16, gait 3.9-3.12, rest tremor 3.17),
`models/` is not required. Tests 3.7, 3.8 and 3.13, and `--tracker rtmpose`,
need it. The program checks before it starts and says which file is missing.

## Usage

```bash
python lidar_cli.py --list-tests                       # what --test accepts
python lidar_cli.py recording.zip --test 3.10          # -> 3.10_Gait.zip
python lidar_cli.py recording.zip --test 3.4a -o results/p07_3.4a.zip
python lidar_cli.py session_folder --test 3.13         # unzipped folder works too
python lidar_cli.py recording.zip --test 3.10 --tracker rtmpose
python lidar_cli.py recording.zip --test 3.17 --no-videos   # CSVs only, faster
```

| Website | Command line |
|---|---|
| Choose a recording | `INPUT`: a `.zip`, or an unzipped session folder |
| MDS-UPDRS Part 3 test menu | `--test ID`, for example `3.10` or `3.4a` |
| Tracker (MediaPipe / RTMPose) | `--tracker mediapipe` (default) or `--tracker rtmpose`. Gait tests only. |
| Download results | `--output PATH` (default: named after the test, in the current folder) |

Other options:

| Option | What it does |
|---|---|
| `--pipeline KIND` | Choose a pipeline directly instead of a test: `auto`, `torso`, `hand`, `foot`, `toe_tap`, `rest_tremor`, `spine` |
| `--no-videos` | Skip the three videos. Much faster; the CSVs are identical. |
| `--graphs` | Also add PNG plots of every angle and joint over time |
| `--overwrite` | Replace an existing output file |
| `--models-dir DIR` | Model folder, if not `models/` (or set `LIDAR_MODELS_DIR`) |
| `--timeout SECONDS` | Give up after this long (default 3000) |
| `--json` | Print a JSON summary, for use in batch scripts |
| `-q` / `-v` | Quiet / show the pipeline's full output |

The program prints the output path on success. Exit codes: 0 = done,
1 = processing failed, 2 = bad input, 3 = model files missing.

### Processing many recordings

```bash
# Windows PowerShell
Get-ChildItem recordings\*.zip | ForEach-Object {
  python lidar_cli.py $_.FullName --test 3.10 -o "results\$($_.BaseName)_3.10.zip" }
```

## Privacy

The `*_rgb.mp4` video in each output, and the recording itself, show the
participant's face. Handle them as identifiable patient data. Use
`--no-videos` for a CSV-only output that holds no images.
