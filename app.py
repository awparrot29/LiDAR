"""
LiDAR Gait Analysis — Web App
Run: python app.py   (activate the lidar-gait-analysis conda env first)
Open: http://localhost:5000
"""
import collections
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import zipfile
from datetime import datetime

from flask import Flask, jsonify, request, send_file

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024  # 2 GB

GAIT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gait-analysis')
MOTION_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'motion-analysis')
PIPELINE_TIMEOUT = 3000  # seconds
MOVIE_NAME = 'gait_skeleton_3d.mp4'
# Raw camera and raw LiDAR, rendered beside the skeleton so a bad result can be
# traced to the sensor rather than the tracker. Working filenames only — what
# the user downloads is named from the chosen test by _output_names().
RGB_NAME = 'preview_rgb.mp4'
DEPTH_NAME = 'preview_lidar.mp4'

# Every .py that must reach the server for the motion-analysis pipelines to run.
# A deploy that drops any of these still passes a naive import check, because
# most are pulled in transitively — extract.py imports background/sessiongeom at
# module load, and skeleton3d is only touched at the final render step. Keeping
# the list here lets /healthz and the deploy guard check the same thing.
MOTION_MODULES = ('angles', 'background', 'csvout', 'depthsmooth', 'detect',
                  'extract', 'preview', 'process_session', 'profiles',
                  'sessiongeom', 'skeleton3d')

# Subject kinds profiles.get() must answer for.
PROFILE_KINDS = ('torso', 'hand', 'foot', 'toe_tap', 'rest_tremor', 'spine')

# MDS-UPDRS test ID -> pipeline kind. Module level so /upload and the admin
# listing agree, and so a stored recording can be labelled with the test it was
# actually recorded for.
TEST_TO_KIND = {
    '3.4a': 'hand',  '3.4b': 'hand',
    '3.5a': 'hand',  '3.5b': 'hand',
    '3.6a': 'hand',  '3.6b': 'hand',
    # 3.7 (Toe Tapping): lower-leg-only profile, 13-frame pixel smoothing.
    # 3.8 (Leg Agility): full-body foot profile for wider context.
    '3.7a': 'toe_tap', '3.7b': 'toe_tap',
    '3.8a': 'foot',    '3.8b': 'foot',
    '3.9':  'torso', '3.10': 'torso', '3.11': 'torso',
    '3.12': 'torso', '3.13': 'spine',
    '3.15a': 'hand', '3.15b': 'hand',
    '3.16a': 'hand', '3.16b': 'hand',
    # 3.17 (Rest Tremor): all four limbs in one recording, 3-frame pixel smoothing.
    '3.17': 'rest_tremor',
}

TEST_LABELS = {
    '3.4a': 'Finger Tapping – Right Hand',
    '3.4b': 'Finger Tapping – Left Hand',
    '3.5a': 'Hand Movements – Right Hand',
    '3.5b': 'Hand Movements – Left Hand',
    '3.6a': 'Pronation-Supination – Right Hand',
    '3.6b': 'Pronation-Supination – Left Hand',
    '3.7a': 'Toe Tapping – Right Foot',
    '3.7b': 'Toe Tapping – Left Foot',
    '3.8a': 'Leg Agility – Right Leg',
    '3.8b': 'Leg Agility – Left Leg',
    '3.9':  'Arising from Chair',
    '3.10': 'Gait',
    '3.11': 'Freezing of Gait',
    '3.12': 'Postural Stability',
    '3.13': 'Posture',
    '3.15a': 'Postural Tremor – Right Hand',
    '3.15b': 'Postural Tremor – Left Hand',
    '3.16a': 'Kinetic Tremor – Right Hand',
    '3.16b': 'Kinetic Tremor – Left Hand',
    '3.17': 'Rest Tremor – All Limbs',
}


def _slug(text):
    """Filename-safe, mirroring motion-analysis/csvout.safe().

    Deliberately duplicated rather than imported: putting MOTION_DIR on the web
    worker's sys.path would also expose extract/detect/angles/profiles, whose
    names are generic enough to shadow something else. csvout is the authority
    for how output is named — this is a copy of one small function, and the two
    must be changed together.
    """
    s = str(text).strip()
    s = re.sub(r'[‒-―]', ' ', s)      # figure/en/em dashes
    s = re.sub(r'\s+', '_', s)
    s = re.sub(r'[^\w.\-]', '', s)
    s = re.sub(r'_{2,}', '_', s).strip('_')
    return s or 'unnamed'


def _output_names(test_id):
    """Download names for a job: (zip, skeleton, rgb, depth).

    Built from the dropdown choice so a folder of downloads is self-describing —
    `3.7a_Toe_Tapping_Right_Foot.zip` rather than twenty files all called
    `gait_coordinates.zip`.
    """
    label = TEST_LABELS.get(test_id or '', '')
    base = _slug(f'{test_id} {label}') if test_id else 'results'
    return (f'{base}.zip', f'{base}_skeleton.mp4',
            f'{base}_rgb.mp4', f'{base}_lidar.mp4')


# /home persists across restarts on Azure App Service; everything else is ephemeral.
UPLOADS_DIR = '/home/uploads'
OUTPUTS_DIR = '/home/outputs'
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Set ADMIN_TOKEN as an Azure App Setting to enable the /admin/uploads page.
# Leave unset to disable admin access entirely.
ADMIN_TOKEN = os.environ.get('ADMIN_TOKEN', '')

_jobs: dict = {}
_lock = threading.Lock()


def _set_progress(job_id: str, percent: float, stage: str) -> None:
    with _lock:
        job = _jobs.get(job_id)
        if job is not None:
            job['percent'] = max(0, min(100, round(percent)))
            job['stage'] = stage

# ---------------------------------------------------------------------------
# HTML (single-file app — no templates folder needed)
# ---------------------------------------------------------------------------

HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LiDAR Gait Analysis</title>
<style>
  :root {
    --bg:       #0d1117;
    --surface:  #161b22;
    --border:   #30363d;
    --text:     #e6edf3;
    --muted:    #8b949e;
    --accent:   #388bfd;
    --accent-lo:#1f4080;
    --success:  #3fb950;
    --error:    #f85149;
    --r:        8px;
  }
  @media (prefers-color-scheme: light) {
    :root {
      --bg:#f6f8fa; --surface:#ffffff; --border:#d0d7de;
      --text:#1f2328; --muted:#656d76; --accent-lo:#ddf4ff;
    }
  }
  :root[data-theme="dark"]  { --bg:#0d1117; --surface:#161b22; --border:#30363d; --text:#e6edf3; --muted:#8b949e; --accent-lo:#1f4080; }
  :root[data-theme="light"] { --bg:#f6f8fa; --surface:#ffffff; --border:#d0d7de; --text:#1f2328; --muted:#656d76; --accent-lo:#ddf4ff; }

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg); color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    min-height: 100vh;
    display: flex; flex-direction: column; align-items: center;
    padding: 3.5rem 1.25rem 4rem;
  }
  .card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--r); padding: 2rem 2rem 1.75rem;
    width: 100%; max-width: 540px;
  }
  header { margin-bottom: 1.75rem; }
  h1 { font-size: 1.35rem; font-weight: 600; letter-spacing: -.01em; }
  .sub { color: var(--muted); font-size: 0.85rem; margin-top: 0.25rem; line-height: 1.5; }

  /* Drop zone */
  .drop {
    border: 2px dashed var(--border); border-radius: var(--r);
    padding: 2.25rem 1.5rem; text-align: center; cursor: pointer;
    transition: border-color .15s, background .15s;
    margin-bottom: 1.125rem;
  }
  .drop:hover, .drop.over { border-color: var(--accent); background: rgba(56,139,253,.05); }
  .drop.has-file { border-color: var(--success); background: rgba(63,185,80,.05); }
  .drop-icon { font-size: 2rem; line-height: 1; margin-bottom: .5rem; }
  .drop-label { font-size: .875rem; color: var(--muted); }
  .drop-label b { color: var(--text); }
  .fname { font-size: .8rem; color: var(--success); margin-top: .4rem; word-break: break-all; }

  .opt-label { font-size: .78rem; letter-spacing: .04em; text-transform: uppercase;
    color: var(--muted); margin-bottom: .4rem; }
  .tracker-row.disabled { opacity: .45; pointer-events: none; }
  /* Tracker selector */
  .tracker-row { display: flex; gap: .625rem; margin-bottom: 1.125rem; }
  .t-opt {
    flex: 1; border: 1px solid var(--border); border-radius: var(--r);
    padding: .6rem .875rem; cursor: pointer; user-select: none;
    transition: border-color .15s, background .15s;
  }
  .t-opt.selected { border-color: var(--accent); background: rgba(56,139,253,.08); }
  .t-opt input { display: none; }
  .t-name { font-size: .8rem; font-weight: 600; }
  .t-desc { font-size: .73rem; color: var(--muted); margin-top: .1rem; }

  /* Buttons */
  .btn-primary {
    width: 100%; padding: .7rem; border: none; border-radius: var(--r);
    font-size: .9rem; font-weight: 600; cursor: pointer;
    background: var(--accent); color: #fff;
    transition: opacity .15s;
  }
  .btn-primary:disabled { opacity: .38; cursor: not-allowed; }
  .btn-primary:hover:not(:disabled) { opacity: .85; }

  /* Status */
  .status {
    margin-top: 1rem; padding: .875rem 1rem;
    border-radius: var(--r); border: 1px solid var(--border);
    font-size: .83rem; display: none; line-height: 1.5;
  }
  .status.vis  { display: block; }
  .status.proc { border-color: var(--accent-lo); background: rgba(56,139,253,.06); }
  .status.done { border-color: #2ea043;           background: rgba(63,185,80,.06); }
  .status.err  { border-color: #6e1f1f;           background: rgba(248,81,73,.06); }
  .spin {
    display: inline-block; width: 12px; height: 12px;
    border: 2px solid var(--accent-lo); border-top-color: var(--accent);
    border-radius: 50%; animation: spin .7s linear infinite;
    vertical-align: middle; margin-right: 5px;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* Progress */
  .pwrap { display: none; margin-top: .7rem; }
  .pwrap.vis { display: block; }
  .ptrack {
    height: 7px; border-radius: 4px; overflow: hidden;
    background: var(--border);
  }
  .pfill {
    height: 100%; width: 0%; border-radius: 4px;
    background: var(--accent);
    transition: width .4s ease;
  }
  .pfill.indet {
    width: 35%;
    animation: slide 1.3s ease-in-out infinite;
  }
  @keyframes slide {
    0%   { margin-left: -35%; }
    100% { margin-left: 100%; }
  }
  .pmeta {
    display: flex; justify-content: space-between; gap: 1rem;
    margin-top: .4rem; font-size: .76rem; color: var(--muted);
  }
  .pmeta .pct { font-variant-numeric: tabular-nums; flex: none; }

  /* Inline player — camera, LiDAR and skeleton side by side */
  .player { display: none; margin-top: 1rem; }
  .player.vis { display: block; }
  .player-h {
    font-size: .78rem; color: var(--muted); margin-bottom: .4rem;
  }
  .player video {
    width: 100%; display: block; border-radius: var(--r);
    border: 1px solid var(--border); background: #000;
  }
  /* auto-fit rather than a fixed 3 columns: a panel that is missing (an older
     job with no previews) closes the gap instead of leaving a hole, and the
     row reflows to stacked on a narrow screen without a breakpoint. */
  .vgrid {
    display: grid; gap: .6rem;
    grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
    align-items: start;
  }
  .vcell { min-width: 0; }
  .vcell .cap {
    font-size: .7rem; color: var(--muted); margin-bottom: .25rem;
    text-align: center; white-space: nowrap; overflow: hidden;
    text-overflow: ellipsis;
  }
  /* The skeleton panel is twice as wide as the two camera panels (it holds two
     views), so let it take two columns when there is room. */
  .vcell.wide { grid-column: span 2; }
  @media (max-width: 560px) { .vcell.wide { grid-column: span 1; } }
  .sync-note {
    font-size: .7rem; color: var(--muted); margin-top: .45rem; text-align: center;
  }

  /* Download */
  .dl-btn {
    display: none; width: 100%; margin-top: .75rem; padding: .7rem;
    border: none; border-radius: var(--r);
    font-size: .9rem; font-weight: 600; cursor: pointer;
    background: var(--success); color: #0d1117;
    text-align: center; text-decoration: none;
  }
  .dl-btn.vis { display: block; }

  /* Hint */
  .hint {
    margin-top: 1.5rem; padding: .875rem 1rem;
    border-radius: var(--r); border: 1px solid var(--border);
    background: rgba(255,255,255,.02);
    font-size: .775rem; color: var(--muted); line-height: 1.65;
  }
  .hint b { color: var(--text); }
  code {
    font-family: ui-monospace, "Cascadia Code", monospace;
    font-size: .85em; background: rgba(255,255,255,.06);
    padding: .1em .3em; border-radius: 3px;
  }

  /* MDS-UPDRS test dropdown */
  .test-select {
    width: 100%; padding: .65rem .875rem; margin-bottom: .5rem;
    background: var(--surface); color: var(--text);
    border: 1px solid var(--border); border-radius: var(--r);
    font-size: .875rem; cursor: pointer;
    -webkit-appearance: none; -moz-appearance: none; appearance: none;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6' viewBox='0 0 10 6'%3E%3Cpath fill='%238b949e' d='M0 0l5 6 5-6z'/%3E%3C/svg%3E");
    background-repeat: no-repeat; background-position: right .875rem center;
  }
  .test-select:focus { outline: none; border-color: var(--accent); box-shadow: 0 0 0 3px rgba(56,139,253,.15); }
  .test-select option, .test-select optgroup { background: var(--surface); color: var(--text); }
  .test-desc {
    padding: .5rem .875rem .65rem; margin-bottom: 1.125rem;
    border-radius: var(--r); border: 1px solid var(--border);
    background: rgba(255,255,255,.02);
    font-size: .78rem; color: var(--muted); line-height: 1.55;
    display: none;
  }
  .test-desc.vis { display: block; }
  .kind-badge {
    display: inline-block; font-size: .7rem; font-weight: 600;
    padding: .1rem .4rem; border-radius: 3px; margin-bottom: .35rem;
    letter-spacing: .04em; text-transform: uppercase;
  }
  .badge-hand  { background: rgba(139,148,158,.15); color: var(--muted); }
  .badge-torso { background: rgba(56,139,253,.15);  color: var(--accent); }
  .badge-foot  { background: rgba(63,185,80,.15);   color: var(--success); }
  .badge-spine { background: rgba(210,120,20,.15);  color: #d27814; }
</style>
</head>
<body>
<div class="card">
  <header>
    <h1>LiDAR Gait Analysis</h1>
    <p class="sub">Upload a Stray Scanner session to extract 3-D joint coordinates</p>
  </header>

  <!-- Upload zone -->
  <div class="drop" id="drop">
    <div class="drop-icon">📦</div>
    <div class="drop-label"><b>Drop your session ZIP here</b><br>or click to browse</div>
    <div class="fname" id="fname"></div>
    <input type="file" id="file" accept=".zip" style="display:none">
  </div>

  <!-- MDS-UPDRS Part 3 test selector -->
  <div class="opt-label">MDS-UPDRS Part 3 Test</div>
  <select class="test-select" id="test-select">
    <option value="" disabled selected>— Select a test —</option>
    <optgroup label="Upper Extremity – Hand Close-Up">
      <option value="3.4a"  data-kind="hand">3.4a · Finger Tapping – Right Hand</option>
      <option value="3.4b"  data-kind="hand">3.4b · Finger Tapping – Left Hand</option>
      <option value="3.5a"  data-kind="hand">3.5a · Hand Movements – Right Hand</option>
      <option value="3.5b"  data-kind="hand">3.5b · Hand Movements – Left Hand</option>
      <option value="3.6a"  data-kind="hand">3.6a · Pronation-Supination – Right Hand</option>
      <option value="3.6b"  data-kind="hand">3.6b · Pronation-Supination – Left Hand</option>
      <option value="3.15a" data-kind="hand">3.15a · Postural Tremor – Right Hand</option>
      <option value="3.15b" data-kind="hand">3.15b · Postural Tremor – Left Hand</option>
      <option value="3.16a" data-kind="hand">3.16a · Kinetic Tremor – Right Hand</option>
      <option value="3.16b" data-kind="hand">3.16b · Kinetic Tremor – Left Hand</option>
    </optgroup>
    <optgroup label="Lower Extremity – Full Body">
      <option value="3.7a"  data-kind="foot">3.7a · Toe Tapping – Right Foot</option>
      <option value="3.7b"  data-kind="foot">3.7b · Toe Tapping – Left Foot</option>
      <option value="3.8a"  data-kind="foot">3.8a · Leg Agility – Right Leg</option>
      <option value="3.8b"  data-kind="foot">3.8b · Leg Agility – Left Leg</option>
    </optgroup>
    <optgroup label="Whole Body – Full Body">
      <option value="3.9"   data-kind="torso">3.9 · Arising from Chair</option>
      <option value="3.10"  data-kind="torso">3.10 · Gait</option>
      <option value="3.11"  data-kind="torso">3.11 · Freezing of Gait</option>
      <option value="3.12"  data-kind="torso">3.12 · Postural Stability</option>
      <option value="3.13"  data-kind="spine">3.13 · Posture</option>
      <option value="3.17"  data-kind="rest_tremor">3.17 · Rest Tremor – All Limbs</option>
    </optgroup>
  </select>
  <div class="test-desc" id="test-desc"></div>

  <!-- Tracker choice (torso only) -->
  <div class="opt-label" id="tracker-label">Tracker</div>
  <div class="tracker-row" id="tracker-row">
    <label class="t-opt selected" id="lbl-mp">
      <input type="radio" name="tracker" value="mediapipe" checked>
      <div class="t-name">MediaPipe</div>
      <div class="t-desc">Faster · runs on CPU</div>
    </label>
    <label class="t-opt" id="lbl-rtp">
      <input type="radio" name="tracker" value="rtmpose">
      <div class="t-name">RTMPose</div>
      <div class="t-desc">More accurate · slower</div>
    </label>
  </div>

  <button class="btn-primary" id="go" disabled>Analyze</button>

  <div class="status" id="st">
    <span class="spin" id="spin"></span><span id="stmsg"></span>
    <div class="pwrap" id="pwrap">
      <div class="ptrack"><div class="pfill" id="pfill"></div></div>
      <div class="pmeta"><span id="pstage"></span><span class="pct" id="ppct"></span></div>
    </div>
  </div>
  <div class="player" id="player">
    <div class="player-h">Camera, LiDAR and tracker &mdash; same frames, same orientation</div>
    <div class="vgrid">
      <div class="vcell" id="cell-rgb">
        <div class="cap">Camera (RGB)</div>
        <video id="vid-rgb" loop muted playsinline preload="auto"></video>
      </div>
      <div class="vcell" id="cell-depth">
        <div class="cap">LiDAR depth &mdash; near blue, far red</div>
        <video id="vid-depth" loop muted playsinline preload="auto"></video>
      </div>
      <div class="vcell wide" id="cell-skel">
        <div class="cap">3D skeleton &mdash; front and side view</div>
        <video id="vid" controls autoplay loop muted playsinline preload="auto"></video>
      </div>
    </div>
    <div class="sync-note">Use the controls on the skeleton &mdash; all three play together.</div>
  </div>

  <a class="dl-btn" id="mdl" download>&#8595; Download 3D Movie (MP4)</a>
  <a class="dl-btn" id="dl">&#8595; Download Results (ZIP: CSVs + 3D movie)</a>

  <div class="hint">
    <b>What to upload:</b> ZIP the output folder from the iPad app <b>Stray Scanner</b>.
    In the Files app, find your recording session folder, long-press it, and tap
    <em>Compress</em> to create a ZIP. The folder must contain
    <code>rgb.mp4</code>, <code>camera_matrix.csv</code>,
    and the <code>depth/</code> &amp; <code>confidence/</code> frame folders.<br><br>
    <b>Select the MDS-UPDRS Part 3 test</b> that matches your recording.
    <em>Upper extremity</em> tests (finger tapping, hand movements, pronation-supination,
    postural/kinetic tremor) use a hand close-up and track 21 finger joints.
    <em>Lower extremity and whole-body</em> tests (gait, arising from chair, toe tapping, leg
    agility, postural stability, posture) use a full-body view and track 12 joints
    (shoulders through ankles).
    <em>Rest tremor (3.17)</em> uses the same full-body view — one recording covers all four
    limbs simultaneously; score each limb from its wrist or ankle CSV.<br><br>
    <b>Output:</b> One CSV per joint with X&nbsp;Y&nbsp;Z coordinates per frame,
    angle CSVs for each measured joint pair, and three movies — the
    <b>camera</b>, the <b>LiDAR depth</b>, and a <b>3D stick figure</b> with bones
    drawn between the joints. Every file is named for the test you chose
    (<code>3.7a_left_knee.csv</code>), and every column carries its unit in the
    header row: <code>x_m</code>, <code>y_m</code>, <code>z_m</code>,
    <code>angle_deg</code>. Fingertip files also carry a <code>time_s</code>
    column. All axes are real-world distances from the camera.<br><br>
    <b>Note:</b> Processing takes 5&nbsp;–&nbsp;15&nbsp;minutes depending on video length.
    Keep this tab open while it runs.
  </div>
</div>

<script>
const drop = document.getElementById('drop');
const file = document.getElementById('file');
const fname = document.getElementById('fname');
const go   = document.getElementById('go');
const st   = document.getElementById('st');
const spin = document.getElementById('spin');
const stmsg = document.getElementById('stmsg');
const pwrap = document.getElementById('pwrap');
const pfill = document.getElementById('pfill');
const pstage = document.getElementById('pstage');
const ppct = document.getElementById('ppct');
const dl   = document.getElementById('dl');
const mdl  = document.getElementById('mdl');
const player = document.getElementById('player');
const vid  = document.getElementById('vid');
const vidRgb = document.getElementById('vid-rgb');
const vidDepth = document.getElementById('vid-depth');
const cellRgb = document.getElementById('cell-rgb');
const cellDepth = document.getElementById('cell-depth');

// The skeleton is the only panel with visible controls; the other two follow it.
// Guarded by `syncing` because assigning currentTime fires another seek event,
// and without the guard the three videos chase each other indefinitely.
let syncing = false;
function followers() {
  return [vidRgb, vidDepth].filter(v => v.getAttribute('src'));
}
function mirror(fn) {
  if (syncing) return;
  syncing = true;
  try { followers().forEach(fn); } finally { syncing = false; }
}
vid.addEventListener('play',  () => mirror(v => v.play().catch(() => {})));
vid.addEventListener('pause', () => mirror(v => v.pause()));
vid.addEventListener('seeking', () => mirror(v => { v.currentTime = vid.currentTime; }));
// Drift correction: the three decoders do not advance in lockstep, so nudge any
// follower that has slipped more than ~2 frames at 60fps.
vid.addEventListener('timeupdate', () => mirror(v => {
  if (Math.abs(v.currentTime - vid.currentTime) > 0.04) v.currentTime = vid.currentTime;
}));

let chosen = null;

drop.addEventListener('click', () => file.click());
drop.addEventListener('dragover', e => { e.preventDefault(); drop.classList.add('over'); });
drop.addEventListener('dragleave', () => drop.classList.remove('over'));
drop.addEventListener('drop', e => {
  e.preventDefault(); drop.classList.remove('over');
  const f = e.dataTransfer.files[0];
  if (f && f.name.toLowerCase().endsWith('.zip')) pick(f);
});
file.addEventListener('change', () => { if (file.files[0]) pick(file.files[0]); });

function pick(f) {
  chosen = f;
  fname.textContent = f.name + '  (' + (f.size / 1048576).toFixed(1) + ' MB)';
  drop.classList.add('has-file');
  syncTracker();
  reset();
}

document.querySelectorAll('.t-opt').forEach(o => o.addEventListener('click', () => {
  const row = o.closest('.tracker-row');
  row.querySelectorAll('.t-opt').forEach(x => x.classList.remove('selected'));
  o.classList.add('selected');
}));

const testSelect = document.getElementById('test-select');
const testDesc   = document.getElementById('test-desc');

const TEST_META = {
  '3.4a':  {kind:'hand',  desc:'Tap index finger to thumb as fast as possible for ~10 sec. Records speed, amplitude, and rhythm decay of each tap.'},
  '3.4b':  {kind:'hand',  desc:'Tap index finger to thumb as fast as possible for ~10 sec. Records speed, amplitude, and rhythm decay of each tap.'},
  '3.5a':  {kind:'hand',  desc:'Open and close the fist fully as fast as possible for ~10 sec. Records amplitude and speed of hand opening.'},
  '3.5b':  {kind:'hand',  desc:'Open and close the fist fully as fast as possible for ~10 sec. Records amplitude and speed of hand opening.'},
  '3.6a':  {kind:'hand',  desc:'Rapidly alternate palm-up and palm-down for ~10 sec. Records rotational speed and regularity.'},
  '3.6b':  {kind:'hand',  desc:'Rapidly alternate palm-up and palm-down for ~10 sec. Records rotational speed and regularity.'},
  '3.7a':  {kind:'foot',  desc:'While seated, tap the right toe up and down repeatedly for ~10 sec. Records cadence, lift amplitude (ankle dorsiflexion), and hesitations.'},
  '3.7b':  {kind:'foot',  desc:'While seated, tap the left toe up and down repeatedly for ~10 sec. Records cadence, lift amplitude (ankle dorsiflexion), and hesitations.'},
  '3.8a':  {kind:'foot',  desc:'While seated, stamp the right foot rapidly for ~10 sec. Records lift height, speed, and regularity. Tracks heel and toe separately.'},
  '3.8b':  {kind:'foot',  desc:'While seated, stamp the left foot rapidly for ~10 sec. Records lift height, speed, and regularity. Tracks heel and toe separately.'},
  '3.9':   {kind:'torso', desc:'Rise from an armless chair with arms crossed over the chest. Records trunk lean, rise speed, and balance on completion.'},
  '3.10':  {kind:'torso', desc:'Walk ~10 m, turn, and return. Records step length, cadence, arm swing, and trunk posture throughout.'},
  '3.11':  {kind:'torso', desc:'Walk through a doorway and turn twice. Detects hesitation, shuffling, and freezing episodes.'},
  '3.12':  {kind:'torso', desc:'Stand and recover from an unexpected backward pull on the shoulders. Records trunk displacement and recovery time.'},
  '3.13':  {kind:'spine', desc:'Stand naturally with eyes open. Records vertebral-level angles (lumbar through cervical) using the SpinePose 37-keypoint model to quantify stooped posture.'},
  '3.15a': {kind:'hand',  desc:'Hold the right arm outstretched and still for ~10 sec. Records involuntary oscillation amplitude and frequency.'},
  '3.15b': {kind:'hand',  desc:'Hold the left arm outstretched and still for ~10 sec. Records involuntary oscillation amplitude and frequency.'},
  '3.16a': {kind:'hand',  desc:'Move the right index finger repeatedly from your nose to a fixed target. Records tremor amplitude during intentional movement.'},
  '3.16b': {kind:'hand',  desc:'Move the left index finger repeatedly from your nose to a fixed target. Records tremor amplitude during intentional movement.'},
  '3.17':  {kind:'rest_tremor', desc:'Sit quietly with hands placed on the chair arms and feet flat on the floor for 10 sec. Records wrist and ankle positions from all four limbs simultaneously — use the output CSVs to score RUE, LUE, RLE, and LLE rest tremor separately.'},
};

testSelect.addEventListener('change', syncTracker);

// RTMPose only exists for the torso pipeline, so grey it out for hand tests.
function syncTracker() {
  const val  = testSelect.value;
  const meta = TEST_META[val];
  const opt  = testSelect.options[testSelect.selectedIndex];
  const kind = (opt && opt.dataset) ? opt.dataset.kind : '';

  if (meta) {
    const badge = kind === 'hand'
      ? '<span class="kind-badge badge-hand">Hand close-up · 21 joints</span>'
      : kind === 'foot'
      ? '<span class="kind-badge badge-foot">Full body · 18 joints (feet)</span>'
      : kind === 'spine'
      ? '<span class="kind-badge badge-spine">Spine · 6 vertebral angles</span>'
      : kind === 'rest_tremor'
      ? '<span class="kind-badge badge-torso">Full body · 12 joints · 3-frame smoothing</span>'
      : '<span class="kind-badge badge-torso">Full body · 12 joints</span>';
    testDesc.innerHTML = badge + '<br>' + meta.desc;
    testDesc.className = 'test-desc vis';
  } else {
    testDesc.className = 'test-desc';
  }

  const trackerRow = document.getElementById('tracker-row');
  const trackerLbl = document.getElementById('tracker-label');
  const off = (kind === 'hand' || kind === 'foot' || kind === 'spine' || kind === 'rest_tremor');
  trackerRow.classList.toggle('disabled', off);
  trackerLbl.textContent = kind === 'hand'        ? 'Tracker (torso only)'
                         : kind === 'foot'        ? 'Tracker (WholeBody, fixed)'
                         : kind === 'spine'       ? 'Tracker (SpinePose, fixed)'
                         : kind === 'rest_tremor' ? 'Tracker (MediaPipe, fixed)'
                         : 'Tracker';

  go.disabled = !(chosen && val);
}
syncTracker();

go.addEventListener('click', async () => {
  if (!chosen || !testSelect.value) return;
  const tracker  = document.querySelector('input[name="tracker"]:checked').value;
  const test     = testSelect.value;
  const testName = testSelect.options[testSelect.selectedIndex].text.split('·').slice(1).join('·').trim();
  go.disabled = true;
  setStatus('proc', 'Uploading…');
  dl.className = 'dl-btn';

  const form = new FormData();
  form.append('session', chosen);
  form.append('tracker', tracker);
  form.append('test', test);

  let jobId;
  try {
    const r = await fetch('/upload', { method: 'POST', body: form });
    const d = await r.json();
    if (!r.ok) throw new Error(d.error || 'Upload failed');
    jobId = d.job_id;
  } catch (e) { return fail(e.message); }

  setStatus('proc', 'Analyzing ' + testName + '…');
  pwrap.className = 'pwrap vis';
  setProgress(0, 'Starting…');
  const started = Date.now();

  const iv = setInterval(async () => {
    try {
      const r = await fetch('/status/' + jobId);
      const d = await r.json();

      if (d.status === 'done') {
        clearInterval(iv);
        spin.style.display = 'none';
        pwrap.className = 'pwrap';
        setStatus('done', d.has_movie
          ? 'Done! The 3D skeleton is playing below.'
          : 'Done! Your coordinates are ready to download.');
        dl.href = '/download/' + jobId;
        dl.className = 'dl-btn vis';
        if (d.has_movie) {
          vid.src = '/movie/' + jobId;
          player.className = 'player vis';
          // ?download=1 makes the server send the same file as an attachment
          mdl.href = '/movie/' + jobId + '?download=1';
          mdl.className = 'dl-btn vis';
          // Camera and LiDAR are best-effort: a session the previews failed on
          // still shows its skeleton rather than an empty row.
          if (d.has_rgb) { vidRgb.src = '/media/' + jobId + '/rgb'; }
          if (d.has_depth) { vidDepth.src = '/media/' + jobId + '/depth'; }
          cellRgb.style.display = d.has_rgb ? '' : 'none';
          cellDepth.style.display = d.has_depth ? '' : 'none';
          // With only the skeleton present there is nothing to sit beside, so
          // let it use the full width instead of two of three columns.
          document.getElementById('cell-skel').className =
            (d.has_rgb || d.has_depth) ? 'vcell wide' : 'vcell';
        }
        go.disabled = false;
      } else if (d.status === 'error') {
        clearInterval(iv);
        pwrap.className = 'pwrap';
        fail(d.error || 'Unknown error');
      } else {
        setProgress(d.percent, d.stage, (Date.now() - started) / 1000);
      }
    } catch (e) { clearInterval(iv); fail('Lost connection to server'); }
  }, 3000);
});

function setProgress(pct, stage, elapsed) {
  // Before the first joint reports in there is nothing to measure, so show a
  // moving stripe rather than a bar frozen at 0%.
  if (!pct) {
    pfill.className = 'pfill indet';
    ppct.textContent = '';
  } else {
    pfill.className = 'pfill';
    pfill.style.width = pct + '%';
    let label = pct + '%';
    if (elapsed && pct >= 3) {
      const left = Math.round(elapsed * (100 - pct) / pct);
      if (left > 0) label += ' · ~' + fmt(left) + ' left';
    }
    ppct.textContent = label;
  }
  pstage.textContent = stage || '';
}
function fmt(s) {
  if (s < 60) return s + 's';
  const m = Math.round(s / 60);
  return m + (m === 1 ? ' min' : ' min');
}

function setStatus(cls, msg) {
  st.className = 'status vis ' + cls;
  spin.style.display = cls === 'proc' ? 'inline-block' : 'none';
  stmsg.textContent = msg;
}
function fail(msg) {
  setStatus('err', 'Error: ' + msg);
  go.disabled = false;
}
function reset() {
  st.className = 'status';
  dl.className = 'dl-btn';
  mdl.className = 'dl-btn';
  pwrap.className = 'pwrap';
  // Stop and unload the previous run's videos, otherwise they keep playing
  // underneath while the next upload is processing
  player.className = 'player';
  [vid, vidRgb, vidDepth].forEach(v => {
    v.pause();
    v.removeAttribute('src');
    v.load();
  });
}
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    return HTML


@app.route('/healthz')
def healthz():
    """Report whether the pipeline's native dependencies import cleanly.

    Runs in a subprocess so a broken native library cannot take down the worker,
    and so it exercises the same interpreter the pipeline itself uses.
    """
    probe = (
        'import json, os, sys; out = {}\n'
        'for mod in ("numpy", "cv2", "mediapipe", "pandas", "matplotlib",\n'
        '            "onnxruntime", "rtmlib", "imageio_ffmpeg"):\n'
        '    try:\n'
        '        m = __import__(mod)\n'
        '        out[mod] = {"ok": True, "version": getattr(m, "__version__", "?"),\n'
        '                    "path": getattr(m, "__file__", "?")}\n'
        '    except Exception as e:\n'
        '        out[mod] = {"ok": False, "error": f"{type(e).__name__}: {e}"}\n'
        'try:\n'
        '    import mediapipe as mp; out["mp.solutions"] = {"ok": hasattr(mp, "solutions")}\n'
        'except Exception as e:\n'
        '    out["mp.solutions"] = {"ok": False, "error": str(e)}\n'
        'try:\n'
        '    from rtmlib import Wholebody; out["rtmlib.Wholebody"] = {"ok": True}\n'
        'except Exception as e:\n'
        '    out["rtmlib.Wholebody"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}\n'
        'try:\n'
        '    from spinepose import PoseTracker as SpinePT, SpinePoseEstimator; out["spinepose"] = {"ok": True}\n'
        'except Exception as e:\n'
        '    out["spinepose"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}\n'
        f'md = {repr(MOTION_DIR)}\n'
        'sys.path.insert(0, md)\n'
        # File presence first: a partial deploy is reported as the missing file
        # it is, rather than as whatever transitive import happens to fail.
        f'expected = {repr(list(MOTION_MODULES))}\n'
        'present = sorted(f[:-3] for f in os.listdir(md) if f.endswith(".py")) if os.path.isdir(md) else []\n'
        'missing = [m for m in expected if m not in present]\n'
        'out["motion.files"] = {"ok": not missing, "dir": md,\n'
        '                       "expected": len(expected), "present": len(present),\n'
        '                       "missing": missing}\n'
        # Then import every pipeline module, not just the ones that happen to be
        # reachable from extract. skeleton3d in particular is only used at the
        # final render step, so an import-time check is the only cheap way to
        # catch it before a full tracking pass has already been spent.
        'for mod in expected:\n'
        '    try:\n'
        '        __import__(mod)\n'
        '        out["motion." + mod] = {"ok": True}\n'
        '    except Exception as e:\n'
        '        out["motion." + mod] = {"ok": False, "error": f"{type(e).__name__}: {e}"}\n'
        # Every kind the upload form can dispatch to, so a profile that raises
        # for one subject cannot hide behind the others.
        f'kinds = {repr(list(PROFILE_KINDS))}\n'
        'for k in kinds:\n'
        '    try:\n'
        '        import profiles; p = profiles.get(k)\n'
        '        out["profiles." + k] = {"ok": True, "model": p["model"],\n'
        '                                "pixel_win": p.get("pixel_smooth_window")}\n'
        '    except Exception as e:\n'
        '        out["profiles." + k] = {"ok": False, "error": f"{type(e).__name__}: {e}"}\n'
        # Single verdict so a deploy can be checked without reading every key.
        'out["ok"] = all(v.get("ok") for v in out.values() if isinstance(v, dict))\n'
        'print(json.dumps(out))\n'
    )
    proc = subprocess.run(
        [sys.executable, '-c', probe], capture_output=True, text=True, timeout=120
    )
    if proc.returncode != 0:
        return jsonify(ok=False, stderr=(proc.stderr or '')[-3000:]), 500
    # 503 on any failed check so a deploy can be gated on the status code alone.
    # Safe to fail loudly here: healthCheckPath is unset and the platform warmup
    # probe hits '/', so a red /healthz cannot stop the container from starting.
    try:
        code = 200 if json.loads(proc.stdout).get('ok') else 503
    except (ValueError, AttributeError):
        code = 500
    return app.response_class(proc.stdout, mimetype='application/json', status=code)


def _meta_path(save_path):
    return save_path + '.meta.json'


def _write_upload_meta(save_path, job_id, test_id, kind, original_name, size):
    """Store the test an upload was recorded for, as a sidecar next to the zip."""
    meta = {
        'job_id': job_id,
        'test': test_id or None,
        'test_label': TEST_LABELS.get(test_id),
        'kind': kind,
        'original_name': original_name,
        'size': size,
        'uploaded_utc': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
        'status': 'processing',
    }
    try:
        with open(_meta_path(save_path), 'w', encoding='utf-8') as fh:
            json.dump(meta, fh, indent=2)
    except OSError:
        pass  # non-fatal — never fail an upload over bookkeeping


def _update_upload_meta(save_path, **fields):
    """Patch an existing sidecar; silently no-op if it was never written."""
    path = _meta_path(save_path)
    try:
        with open(path, encoding='utf-8') as fh:
            meta = json.load(fh)
    except (OSError, ValueError):
        return
    meta.update(fields)
    try:
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump(meta, fh, indent=2)
    except OSError:
        pass


def _read_upload_meta(name):
    """Sidecar for an uploads/ filename, or {} if there is none."""
    try:
        with open(os.path.join(UPLOADS_DIR, name + '.meta.json'), encoding='utf-8') as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


@app.route('/upload', methods=['POST'])
def upload():
    f = request.files.get('session')
    if not f:
        return jsonify(error='No file received'), 400

    tracker = request.form.get('tracker', 'mediapipe')
    if tracker not in ('mediapipe', 'rtmpose'):
        tracker = 'mediapipe'

    # Map the MDS-UPDRS test ID to the pipeline kind (torso/hand).
    # The old 'subject' field is kept as a fallback for direct API callers.
    test_id = request.form.get('test', '')
    if test_id in TEST_TO_KIND:
        subject = TEST_TO_KIND[test_id]
    else:
        subject = request.form.get('subject', 'auto')
        if subject not in ('auto', 'torso', 'hand'):
            subject = 'auto'

    job_id = str(uuid.uuid4())
    zip_bytes = f.read()

    # Persist the raw upload so it can be retrieved later via /admin/uploads.
    safe_name = re.sub(r'[^\w\-.]', '_', f.filename or 'upload')[:80]
    ts = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    save_path = os.path.join(UPLOADS_DIR, f"{ts}_{job_id[:8]}_{safe_name}")
    try:
        with open(save_path, 'wb') as fh:
            fh.write(zip_bytes)
    except OSError:
        pass  # non-fatal; processing continues even if the save fails

    # Record WHICH test this recording was for, next to the recording itself.
    # Without this a stored upload cannot be matched back to its MDS-UPDRS item:
    # the five recordings stranded by a failed batch on 2026-09-24 could only be
    # re-run by guessing, because nothing tied them to a test. A sidecar .json
    # keeps older uploads valid — they simply have no metadata.
    _write_upload_meta(save_path, job_id, test_id, subject, f.filename, len(zip_bytes))

    with _lock:
        _jobs[job_id] = {'status': 'processing', 'result': None, 'error': None,
                         'percent': 0, 'stage': 'Queued…'}

    threading.Thread(target=_run_job,
                     args=(job_id, zip_bytes, tracker, subject, save_path,
                           test_id),
                     daemon=True).start()
    return jsonify(job_id=job_id)


@app.route('/status/<job_id>')
def status(job_id):
    with _lock:
        job = dict(_jobs.get(job_id, {}))
    if not job:
        return jsonify(error='unknown job'), 404
    return jsonify(status=job['status'], error=job.get('error'),
                   percent=job.get('percent', 0), stage=job.get('stage', ''),
                   has_movie=bool(job.get('movie')),
                   has_rgb=bool(job.get('rgb')),
                   has_depth=bool(job.get('depth')))


@app.route('/download/<job_id>')
def download(job_id):
    with _lock:
        job = dict(_jobs.get(job_id, {}))
    if not job or job['status'] != 'done':
        return jsonify(error='result not ready'), 400
    return send_file(
        io.BytesIO(job['result']),
        as_attachment=True,
        download_name=job.get('zip_name') or 'gait_coordinates.zip',
        mimetype='application/zip',
    )


@app.route('/movie/<job_id>')
def movie(job_id):
    """The 3D stick figure MP4.

    Served inline by default so the page can play it in a <video> element;
    ?download=1 switches to an attachment so the same file can be saved without
    unzipping the results bundle.
    """
    with _lock:
        job = dict(_jobs.get(job_id, {}))
    if not job or job.get('status') != 'done' or not job.get('movie'):
        return jsonify(error='movie not available'), 400
    # conditional=True gives range requests, which players use to seek
    return send_file(
        io.BytesIO(job['movie']),
        mimetype='video/mp4',
        download_name=job.get('movie_name') or MOVIE_NAME,
        as_attachment=request.args.get('download') == '1',
        conditional=True,
    )


@app.route('/media/<job_id>/<which>')
def media(job_id, which):
    """The raw camera ('rgb') or raw LiDAR ('depth') movie for a finished job.

    Separate from /movie because these two are best-effort: a session whose
    previews failed still returns its skeleton, and the page simply hides the
    panels it cannot fill.
    """
    if which not in ('rgb', 'depth'):
        return jsonify(error='unknown media'), 404
    with _lock:
        job = dict(_jobs.get(job_id, {}))
    if not job or job.get('status') != 'done' or not job.get(which):
        return jsonify(error=f'{which} not available'), 400
    return send_file(
        io.BytesIO(job[which]),
        mimetype='video/mp4',
        download_name=job.get(f'{which}_name') or f'{which}.mp4',
        as_attachment=request.args.get('download') == '1',
        conditional=True,
    )


# ---------------------------------------------------------------------------
# Admin — upload retrieval
# ---------------------------------------------------------------------------

def _check_admin(req):
    """Return None if the request is authorised, or a (message, status) tuple."""
    if not ADMIN_TOKEN:
        return ('ADMIN_TOKEN app setting is not configured on this server.', 403)
    if req.args.get('token') != ADMIN_TOKEN:
        return ('Invalid or missing token. Append ?token=<ADMIN_TOKEN> to the URL.', 401)
    return None


@app.route('/admin/uploads')
def admin_uploads():
    err = _check_admin(request)
    if err:
        return err

    entries = []
    for name in os.listdir(UPLOADS_DIR):
        path = os.path.join(UPLOADS_DIR, name)
        # Sidecars describe the recording next to them; they are not uploads.
        if not os.path.isfile(path) or name.endswith('.meta.json'):
            continue
        size_kb = os.path.getsize(path) // 1024
        mtime = datetime.utcfromtimestamp(os.path.getmtime(path)).strftime('%Y-%m-%d %H:%M UTC')
        entries.append((name, size_kb, mtime, _read_upload_meta(name)))
    entries.sort(key=lambda e: e[2], reverse=True)

    token = request.args['token']

    def _cells(name, size_kb, mtime, meta):
        # Uploads predating the sidecar have no test recorded, and it cannot be
        # recovered after the fact — say so rather than implying 'none'.
        test = meta.get('test')
        label = meta.get('test_label') or ''
        test_cell = f'{test} · {label}' if test else '<i>not recorded</i>'
        kind = meta.get('kind') or '—'
        status = meta.get('status') or '<i>unknown</i>'
        colour = {'done': '#0a0', 'error': '#c00'}.get(meta.get('status'), '#888')
        err = meta.get('error')
        if err:
            status += f'<br><span style="font-size:11px">{err[:120]}</span>'
        return (f'<tr><td style="padding:4px 12px">{name}</td>'
                f'<td style="padding:4px 12px">{test_cell}</td>'
                f'<td style="padding:4px 12px">{kind}</td>'
                f'<td style="padding:4px 12px;color:{colour}">{status}</td>'
                f'<td style="padding:4px 12px">{size_kb} KB</td>'
                f'<td style="padding:4px 12px">{mtime}</td>'
                f'<td style="padding:4px 12px">'
                f'<a href="/admin/download/{name}?token={token}">download</a></td></tr>')

    rows = ''.join(_cells(*e) for e in entries)
    n_failed = sum(1 for e in entries if e[3].get('status') == 'error')
    banner = (f'<p style="color:#c00">{n_failed} upload(s) recorded as failed.</p>'
              if n_failed else '')
    html = (
        '<html><body style="font-family:monospace">'
        f'<h3>Saved uploads ({len(entries)} files)</h3>'
        + banner +
        '<table border="1" cellspacing="0"><tr>'
        '<th style="padding:4px 12px">File</th>'
        '<th style="padding:4px 12px">Test</th>'
        '<th style="padding:4px 12px">Kind</th>'
        '<th style="padding:4px 12px">Result</th>'
        '<th style="padding:4px 12px">Size</th>'
        '<th style="padding:4px 12px">Uploaded</th>'
        '<th></th></tr>'
        + rows +
        '</table></body></html>'
    )
    return html


@app.route('/admin/outputs')
def admin_outputs():
    err = _check_admin(request)
    if err:
        return err

    entries = []
    for name in os.listdir(OUTPUTS_DIR):
        path = os.path.join(OUTPUTS_DIR, name)
        if not os.path.isfile(path):
            continue
        size_kb = os.path.getsize(path) // 1024
        mtime = datetime.utcfromtimestamp(os.path.getmtime(path)).strftime('%Y-%m-%d %H:%M UTC')
        entries.append((name, size_kb, mtime))
    entries.sort(key=lambda e: e[2], reverse=True)

    token = request.args['token']
    rows = ''.join(
        f'<tr><td style="padding:4px 12px">{name}</td>'
        f'<td style="padding:4px 12px">{size_kb} KB</td>'
        f'<td style="padding:4px 12px">{mtime}</td>'
        f'<td style="padding:4px 12px"><a href="/admin/download-output/{name}?token={token}">download</a></td></tr>'
        for name, size_kb, mtime in entries
    )
    html = (
        '<html><body style="font-family:monospace">'
        f'<h3>Saved outputs ({len(entries)} files)</h3>'
        '<table border="1" cellspacing="0"><tr>'
        '<th style="padding:4px 12px">File</th>'
        '<th style="padding:4px 12px">Size</th>'
        '<th style="padding:4px 12px">Time</th>'
        '<th></th></tr>'
        + rows +
        '</table></body></html>'
    )
    return html


@app.route('/admin/download-output/<path:filename>')
def admin_download_output(filename):
    err = _check_admin(request)
    if err:
        return err

    if '/' in filename or '\\' in filename or '..' in filename:
        return ('Bad filename.', 400)
    path = os.path.join(OUTPUTS_DIR, filename)
    if not os.path.isfile(path):
        return ('File not found.', 404)
    return send_file(path, as_attachment=True, download_name=filename)


@app.route('/admin/download/<path:filename>')
def admin_download(filename):
    err = _check_admin(request)
    if err:
        return err

    # Guard against path traversal
    if '/' in filename or '\\' in filename or '..' in filename:
        return ('Bad filename.', 400)
    path = os.path.join(UPLOADS_DIR, filename)
    if not os.path.isfile(path):
        return ('File not found.', 404)
    return send_file(path, as_attachment=True, download_name=filename)


# ---------------------------------------------------------------------------
# Background processing
# ---------------------------------------------------------------------------

def _preview_src(session: str) -> str:
    """Statements that render the camera and LiDAR movies, appended to a script.

    preview.render_both swallows and reports its own failures, so this stays a
    plain call — the surrounding scripts are '; '-joined one-liners and cannot
    hold a try/except.
    """
    return '; '.join([
        f'sys.path.insert(0, {repr(MOTION_DIR)})',
        'import preview',
        f'preview.render_both({repr(session)}, {repr(RGB_NAME)}, '
        f'{repr(DEPTH_NAME)}, fps=60.0)',
    ])


def _run_job(job_id: str, zip_bytes: bytes, tracker: str,
             subject: str = 'auto', save_path: str = '',
             test_id: str = '') -> None:
    work = tempfile.mkdtemp(prefix='lidar_')
    try:
        # --- Extract ZIP ---
        _set_progress(job_id, 0, 'Extracting ZIP…')
        zpath = os.path.join(work, 'upload.zip')
        with open(zpath, 'wb') as fh:
            fh.write(zip_bytes)
        with zipfile.ZipFile(zpath) as zf:
            _extract_all(zf, work)
        os.remove(zpath)

        # --- Find session folder ---
        session = _find_session(work)
        if session is None:
            raise RuntimeError(
                'No valid Stray Scanner session found in the ZIP. '
                'Expected contents: rgb.mp4, camera_matrix.csv, depth/, confidence/'
            )

        # --- Decide torso or hand ---
        # Run detection in its own process: it loads MediaPipe, which we do not
        # want resident in the web worker. Lower-body visibility separates a
        # walking subject (0.99) from a hand close-up (0.00) cleanly.
        if subject == 'auto':
            _set_progress(job_id, 0, 'Detecting subject…')
            probe_src = '; '.join([
                'import sys, os',
                f'sys.path.insert(0, {repr(MOTION_DIR)})',
                'import detect',
                f'k, ev = detect.detect(os.path.join({repr(work)}, {repr(session)}))',
                'print("@@KIND " + k)',
                'print(detect.explain(ev))',
            ])
            pr = subprocess.run([sys.executable, '-c', probe_src],
                                capture_output=True, text=True, cwd=work)
            kind = 'torso'
            for ln in (pr.stdout or '').splitlines():
                if ln.startswith('@@KIND '):
                    kind = ln[7:].strip()
            if kind not in ('torso', 'hand'):
                kind = 'torso'
            app.logger.info('subject auto-detected as %s; %s', kind,
                            (pr.stdout or '').replace('\n', ' ').strip())
        else:
            kind = subject

        # --- Build pipeline script ---
        # We run calculateangle in a subprocess so CWD is `work` and relative
        # output paths (charts/<session>/data/*.csv) land inside `work`.
        # The tracker swap works by injecting pipelandmark_rtmpose into
        # sys.modules['pipelandmark'] before calculateangle imports it.
        data_rel = os.path.join('charts', session, 'data')
        cam_rel = os.path.join(session, 'camera_matrix.csv')

        if kind in ('hand', 'foot', 'toe_tap', 'rest_tremor', 'spine'):
            # motion-analysis writes data/ under the folder it is given, so
            # pointing it at charts/<session> puts the CSVs exactly where the
            # bundling step below already looks for them.
            out_rel = os.path.join('charts', session)
            script = '; '.join([
                'import sys, os',
                'import matplotlib; matplotlib.use("Agg")',
                f'sys.path.insert(0, {repr(MOTION_DIR)})',
                'import extract, angles, profiles, skeleton3d',
                f'_p = profiles.get({repr(kind)})',
                f'_a, _geom = extract.extract_all_landmarks({repr(session)}, kind={repr(kind)})',
                '_ang = angles.compute(_a, _p)',
                f'angles.write(_a, _ang, _p, {repr(out_rel)}, fps=60.0, '
                f'graphs=True, test_id={repr(test_id or None)})',
                f'skeleton3d.render(_a, _p, {repr(MOVIE_NAME)}, fps=60.0)',
                _preview_src(session),
            ])
        else:
            script = None

        script_lines = [
            'import sys, os',
            'import matplotlib; matplotlib.use("Agg")',  # headless server — no display
            f'sys.path.insert(0, {repr(GAIT_DIR)})',
        ]
        if tracker == 'rtmpose':
            script_lines += [
                'import pipelandmark_rtmpose as _t',
                'sys.modules["pipelandmark"] = _t',
            ]
        script_lines += [
            'import calculateangle',
            f'calculateangle.main(folder={repr(session)}, '
            f'test_id={repr(test_id or None)})',
            # Then the 3D stick figure, built from the CSVs just written.
            # No intrinsics repair here: pipelandmark now derives them from the
            # session's own orientation and camera matrix, so applying
            # skeleton3d.repair_intrinsics would correct an already-correct
            # projection twice.
            'import skeleton3d',
            f'_lm = skeleton3d.load_landmarks({repr(data_rel)}, '
            f'test_id={repr(test_id or None)})',
            f'skeleton3d.render_movie(_lm, {repr(MOVIE_NAME)})',
            _preview_src(session),
        ]
        if script is None:
            script = '; '.join(script_lines)

        # Fail with something actionable rather than an ImportError traceback.
        # The tracker dropdown applies only to the gait-analysis torso pipeline;
        # hand and foot use their own fixed models (MediaPipe Hands and rtmlib
        # Wholebody respectively), so only check rtmlib when torso+rtmpose.
        if tracker == 'rtmpose' and kind == 'torso':
            probe = subprocess.run([sys.executable, '-c', 'import rtmlib'],
                                   capture_output=True, text=True)
            if probe.returncode != 0:
                raise RuntimeError(
                    'The RTMPose tracker is unavailable — rtmlib failed to '
                    'import on the server. Please use the MediaPipe tracker.\n'
                    + (probe.stderr or '')[-500:])

        # Stream the child's output so @@JOINT / @@FRAME markers can drive the
        # progress bar. -u keeps the pipe unbuffered so they arrive live.
        _set_progress(job_id, 0, f'Starting {kind} pose estimation…')
        proc = subprocess.Popen(
            [sys.executable, '-u', '-c', script],
            cwd=work,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        tail = collections.deque(maxlen=100)
        deadline = time.monotonic() + PIPELINE_TIMEOUT
        n_joints, cur_joint = (15 if kind == 'hand' else 8 if kind == 'foot' else 6), 0
        # Four phases share the bar: the single tracking pass over the video,
        # the quick per-joint angle maths, rendering the 3D movie, then the two
        # camera/LiDAR previews. A fully cached session emits no @@FRAME, so the
        # joints carry the first stretch.
        TRACK_SHARE = 55
        ANGLE_SHARE = 60
        RENDER_SHARE = 90
        PREVIEW_MID = 94          # boundary between the rgb and lidar previews
        tracked = False

        for raw in proc.stdout:
            tail.append(raw)
            line = raw.strip()

            if line.startswith('@@FRAME '):
                try:
                    done, total = (int(v) for v in line[8:].split('/'))
                    if total:
                        tracked = True
                        _set_progress(job_id, TRACK_SHARE * done / total,
                                      f'Tracking pose — frame {done}/{total}')
                except ValueError:
                    pass
            elif line.startswith('@@JOINT '):
                try:
                    frac, cur_name = line[8:].split(' ', 1)
                    cur_joint, n_joints = (int(v) for v in frac.split('/'))
                    if tracked:
                        pct = TRACK_SHARE + (ANGLE_SHARE - TRACK_SHARE) \
                            * cur_joint / n_joints
                        _set_progress(job_id, pct,
                                      f'Computing angles — {cur_name} '
                                      f'({cur_joint}/{n_joints})')
                    else:
                        _set_progress(job_id, TRACK_SHARE * (cur_joint - 1) / n_joints,
                                      f'Loading cached joint {cur_joint}/{n_joints} '
                                      f'— {cur_name}')
                except ValueError:
                    pass
            elif line.startswith('@@RENDER '):
                try:
                    done, total = (int(v) for v in line[9:].split('/'))
                    if total:
                        pct = ANGLE_SHARE + (RENDER_SHARE - ANGLE_SHARE) * done / total
                        _set_progress(job_id, pct,
                                      f'Rendering 3D movie — frame {done}/{total}')
                except ValueError:
                    pass
            elif line.startswith('@@PREVIEW '):
                # Two previews share the last stretch: camera then LiDAR.
                try:
                    label, frac = line[10:].split(' ', 1)
                    done, total = (int(v) for v in frac.split('/'))
                    lo = RENDER_SHARE if label == 'rgb' else PREVIEW_MID
                    hi = PREVIEW_MID if label == 'rgb' else 98
                    if total:
                        _set_progress(job_id, lo + (hi - lo) * done / total,
                                      f'Rendering {label} movie — '
                                      f'frame {done}/{total}')
                except ValueError:
                    pass

            if time.monotonic() > deadline:
                proc.kill()
                raise RuntimeError(
                    f'Pipeline timed out after {PIPELINE_TIMEOUT // 60} minutes.')

        if proc.wait() != 0:
            raise RuntimeError('Pipeline failed:\n' + ''.join(tail)[-3000:])

        # --- Bundle CSVs + movie ---
        _set_progress(job_id, 99, 'Bundling results…')
        data_dir = os.path.join(work, 'charts', session, 'data')
        if not os.path.isdir(data_dir):
            raise RuntimeError(
                'Pipeline ran but produced no CSV files. '
                'Check that the video and LiDAR frames are valid.'
            )

        zip_name, skel_name, rgb_name, depth_name = _output_names(test_id)

        def _read(fname):
            path = os.path.join(work, fname)
            if not os.path.exists(path):
                return None
            with open(path, 'rb') as fh:
                return fh.read()

        movie = _read(MOVIE_NAME)
        rgb = _read(RGB_NAME)
        depth = _read(DEPTH_NAME)

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
            for name in sorted(os.listdir(data_dir)):
                if name.endswith('.csv'):
                    zf.write(os.path.join(data_dir, name), name)
            # Already-compressed video: storing it avoids a pointless deflate pass
            for blob, name in ((movie, skel_name), (rgb, rgb_name),
                               (depth, depth_name)):
                if blob is not None:
                    zf.writestr(name, blob, compress_type=zipfile.ZIP_STORED)
        buf.seek(0)
        result_bytes = buf.read()

        # Persist output to /home/outputs so it can be retrieved later. The test
        # id goes in the stored name too, so a folder of past runs can be read
        # without opening each zip.
        ts = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        out_stem = f"{ts}_{job_id[:8]}"
        if test_id:
            out_stem += f"_{_slug(test_id)}"
        try:
            with open(os.path.join(OUTPUTS_DIR, f"{out_stem}_output.zip"), 'wb') as fh:
                fh.write(result_bytes)
            if movie is not None:
                with open(os.path.join(OUTPUTS_DIR, f"{out_stem}_skeleton.mp4"), 'wb') as fh:
                    fh.write(movie)
        except OSError:
            pass

        with _lock:
            _jobs[job_id] = {'status': 'done', 'result': result_bytes, 'error': None,
                             'percent': 100, 'stage': 'Complete',
                             'movie': movie, 'rgb': rgb, 'depth': depth,
                             'zip_name': zip_name, 'movie_name': skel_name,
                             'rgb_name': rgb_name, 'depth_name': depth_name}
        # Outcome goes on the stored recording, not just in memory. _jobs is
        # wiped on restart, so without this a batch that failed weeks ago is
        # invisible — which is exactly how the 2026-09-24 failures went
        # unnoticed until someone went looking.
        if save_path:
            _update_upload_meta(save_path, status='done', error=None)

    except Exception as exc:
        with _lock:
            prev = _jobs.get(job_id, {})
            _jobs[job_id] = {'status': 'error', 'result': None, 'error': str(exc),
                             'percent': prev.get('percent', 0),
                             'stage': prev.get('stage', '')}
        if save_path:
            _update_upload_meta(save_path, status='error', error=str(exc)[:500])
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _extract_all(zf: zipfile.ZipFile, dest_root: str) -> None:
    """Extract every member, tolerating Windows-style backslash entry names.

    Some Windows zip tools (notably .NET's ZipFile.CreateFromDirectory) store
    entries as `depth\\000000.png`. The stdlib treats that as a filename rather
    than a path, so the depth/ folder never appears and the session looks
    invalid. Normalising the separator here keeps those archives usable.
    Path components are filtered so a crafted entry cannot escape dest_root.
    """
    for info in zf.infolist():
        name = info.filename.replace('\\', '/')
        if name.endswith('/'):
            continue
        parts = [p for p in name.split('/') if p not in ('', '.', '..')]
        if not parts:
            continue
        dest = os.path.join(dest_root, *parts)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with zf.open(info) as src, open(dest, 'wb') as dst:
            shutil.copyfileobj(src, dst)


def _find_session(root: str):
    """Return the relative path (from root) to the first valid Stray Scanner session.

    A valid session folder contains rgb.mp4, camera_matrix.csv, and a depth/ subfolder.
    Returns None if nothing is found.
    """
    for dirpath, dirs, files in os.walk(root):
        # Skip the shadow tree a Mac/iOS "Compress" adds
        dirs[:] = [d for d in dirs if d != '__MACOSX']
        if 'rgb.mp4' in files and 'camera_matrix.csv' in files and 'depth' in dirs:
            rel = os.path.relpath(dirpath, root)
            return rel
    return None


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
