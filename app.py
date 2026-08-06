"""
LiDAR Gait Analysis — Web App
Run: python app.py   (activate the lidar-gait-analysis conda env first)
Open: http://localhost:5000
"""
import collections
import io
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import zipfile

from flask import Flask, jsonify, request, send_file

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024  # 2 GB

GAIT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gait-analysis')
PIPELINE_TIMEOUT = 3000  # seconds

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

  <!-- Tracker choice -->
  <div class="tracker-row">
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

  <button class="btn-primary" id="go" disabled>Analyze Gait</button>

  <div class="status" id="st">
    <span class="spin" id="spin"></span><span id="stmsg"></span>
    <div class="pwrap" id="pwrap">
      <div class="ptrack"><div class="pfill" id="pfill"></div></div>
      <div class="pmeta"><span id="pstage"></span><span class="pct" id="ppct"></span></div>
    </div>
  </div>
  <a class="dl-btn" id="dl">&#8595; Download Coordinates (ZIP)</a>

  <div class="hint">
    <b>What to upload:</b> ZIP the output folder from the iPad app <b>Stray Scanner</b>.
    In the Files app, find your recording session folder, long-press it, and tap
    <em>Compress</em> to create a ZIP. The folder must contain
    <code>rgb.mp4</code>, <code>camera_matrix.csv</code>,
    and the <code>depth/</code> &amp; <code>confidence/</code> frame folders.<br><br>
    <b>Output:</b> One CSV per joint (ankle, knee, hip, shoulder, elbow, wrist — left &amp; right)
    with X&nbsp;Y&nbsp;Z coordinates per frame, plus angle CSVs for knees, hips, and elbows.<br><br>
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
  go.disabled = false;
  reset();
}

document.querySelectorAll('.t-opt').forEach(o => o.addEventListener('click', () => {
  document.querySelectorAll('.t-opt').forEach(x => x.classList.remove('selected'));
  o.classList.add('selected');
}));

go.addEventListener('click', async () => {
  if (!chosen) return;
  const tracker = document.querySelector('input[name="tracker"]:checked').value;
  go.disabled = true;
  setStatus('proc', 'Uploading…');
  dl.className = 'dl-btn';

  const form = new FormData();
  form.append('session', chosen);
  form.append('tracker', tracker);

  let jobId;
  try {
    const r = await fetch('/upload', { method: 'POST', body: form });
    const d = await r.json();
    if (!r.ok) throw new Error(d.error || 'Upload failed');
    jobId = d.job_id;
  } catch (e) { return fail(e.message); }

  setStatus('proc', 'Analyzing gait…');
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
        setStatus('done', 'Done! Your coordinates are ready to download.');
        dl.href = '/download/' + jobId;
        dl.className = 'dl-btn vis';
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
  pwrap.className = 'pwrap';
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
        'import json, sys; out = {}\n'
        'for mod in ("numpy", "cv2", "mediapipe", "pandas", "matplotlib",\n'
        '            "onnxruntime", "rtmlib"):\n'
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
        'print(json.dumps(out))\n'
    )
    proc = subprocess.run(
        [sys.executable, '-c', probe], capture_output=True, text=True, timeout=120
    )
    if proc.returncode != 0:
        return jsonify(ok=False, stderr=(proc.stderr or '')[-3000:]), 500
    return app.response_class(proc.stdout, mimetype='application/json')


@app.route('/upload', methods=['POST'])
def upload():
    f = request.files.get('session')
    if not f:
        return jsonify(error='No file received'), 400

    tracker = request.form.get('tracker', 'mediapipe')
    if tracker not in ('mediapipe', 'rtmpose'):
        tracker = 'mediapipe'

    job_id = str(uuid.uuid4())
    zip_bytes = f.read()

    with _lock:
        _jobs[job_id] = {'status': 'processing', 'result': None, 'error': None,
                         'percent': 0, 'stage': 'Queued…'}

    threading.Thread(target=_run_job, args=(job_id, zip_bytes, tracker), daemon=True).start()
    return jsonify(job_id=job_id)


@app.route('/status/<job_id>')
def status(job_id):
    with _lock:
        job = dict(_jobs.get(job_id, {}))
    if not job:
        return jsonify(error='unknown job'), 404
    return jsonify(status=job['status'], error=job.get('error'),
                   percent=job.get('percent', 0), stage=job.get('stage', ''))


@app.route('/download/<job_id>')
def download(job_id):
    with _lock:
        job = dict(_jobs.get(job_id, {}))
    if not job or job['status'] != 'done':
        return jsonify(error='result not ready'), 400
    return send_file(
        io.BytesIO(job['result']),
        as_attachment=True,
        download_name='gait_coordinates.zip',
        mimetype='application/zip',
    )


# ---------------------------------------------------------------------------
# Background processing
# ---------------------------------------------------------------------------

def _run_job(job_id: str, zip_bytes: bytes, tracker: str) -> None:
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

        # --- Build pipeline script ---
        # We run calculateangle in a subprocess so CWD is `work` and relative
        # output paths (charts/<session>/data/*.csv) land inside `work`.
        # The tracker swap works by injecting pipelandmark_rtmpose into
        # sys.modules['pipelandmark'] before calculateangle imports it.
        script_lines = [
            'import sys',
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
            f'calculateangle.main(folder={repr(session)})',
        ]
        script = '; '.join(script_lines)

        # Fail with something actionable rather than an ImportError traceback.
        if tracker == 'rtmpose':
            probe = subprocess.run([sys.executable, '-c', 'import rtmlib'],
                                   capture_output=True, text=True)
            if probe.returncode != 0:
                raise RuntimeError(
                    'The RTMPose tracker is unavailable — rtmlib failed to '
                    'import on the server. Please use the MediaPipe tracker.\n'
                    + (probe.stderr or '')[-500:])

        # Stream the child's output so @@JOINT / @@FRAME markers can drive the
        # progress bar. -u keeps the pipe unbuffered so they arrive live.
        _set_progress(job_id, 0, 'Starting pose estimation…')
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
        n_joints, cur_joint = 6, 0
        # Tracking is a single pass over the video and dominates the runtime, so
        # it owns most of the bar; the per-joint angle maths afterwards is quick.
        # A fully cached session emits no @@FRAME at all, so the joints become
        # the only thing left to measure.
        TRACK_SHARE = 92
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
                        pct = TRACK_SHARE + (100 - TRACK_SHARE) * cur_joint / n_joints
                        _set_progress(job_id, pct,
                                      f'Computing angles — {cur_name} '
                                      f'({cur_joint}/{n_joints})')
                    else:
                        _set_progress(job_id, 100 * (cur_joint - 1) / n_joints,
                                      f'Loading cached joint {cur_joint}/{n_joints} '
                                      f'— {cur_name}')
                except ValueError:
                    pass

            if time.monotonic() > deadline:
                proc.kill()
                raise RuntimeError(
                    f'Pipeline timed out after {PIPELINE_TIMEOUT // 60} minutes.')

        if proc.wait() != 0:
            raise RuntimeError('Pipeline failed:\n' + ''.join(tail)[-3000:])

        # --- Bundle CSVs ---
        _set_progress(job_id, 99, 'Bundling CSVs…')
        data_dir = os.path.join(work, 'charts', session, 'data')
        if not os.path.isdir(data_dir):
            raise RuntimeError(
                'Pipeline ran but produced no CSV files. '
                'Check that the video and LiDAR frames are valid.'
            )

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
            for name in sorted(os.listdir(data_dir)):
                if name.endswith('.csv'):
                    zf.write(os.path.join(data_dir, name), name)
        buf.seek(0)

        with _lock:
            _jobs[job_id] = {'status': 'done', 'result': buf.read(), 'error': None,
                             'percent': 100, 'stage': 'Complete'}

    except Exception as exc:
        with _lock:
            prev = _jobs.get(job_id, {})
            _jobs[job_id] = {'status': 'error', 'result': None, 'error': str(exc),
                             'percent': prev.get('percent', 0),
                             'stage': prev.get('stage', '')}
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
