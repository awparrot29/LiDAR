#!/usr/bin/env python3
"""Build, verify, and ship the LiDar web app — with a completeness gate.

Why this exists: a deploy once reached Azure carrying only 5 of the 9
motion-analysis modules. Every non-torso pipeline (hand, foot, toe_tap,
rest_tremor, spine) died with `ModuleNotFoundError: No module named
'background'`, because extract.py imports it at module load. Nothing in the
build noticed, so the break only surfaced as a failed upload.

Two structural guards prevent a repeat:

  * The staging set is an ALLOWLIST, not "copy the tree and exclude junk".
    The repo root holds ~200 MB of .mp4/.zip/.png scratch, so a denylist is one
    forgotten extension away from a bloated or broken zip. An allowlist can
    only ship what is named.
  * Every module in app.MOTION_MODULES must be present AND byte-identical to
    the repo copy — checked in staging, then re-checked by reading the built
    zip back. A partial zip cannot reach Azure.

Notes on the platform, learned the hard way:
  * `az webapp up` fails on this subscription's policy — always `az webapp
    deploy --type zip`.
  * Build the zip with Python's zipfile (forward slashes). PowerShell's
    Compress-Archive writes backslash paths that Linux extraction mangles.
  * Deploy with --async true AND --track-status false, then poll. Holding the
    HTTP connection open just collects a 504 while Oryx keeps building
    server-side, and --track-status (on by default) makes the CLI declare
    DEPLOYMENT FAILED when the ~3 GB tarball takes over 10 minutes to extract
    on startup — for a build that in fact succeeded.
  * A successful build does NOT mean the new code is being served. The running
    container keeps using the extraction it booted with until restarted.

Usage:
    python deploy.py            # verify, build, deploy, gate on /healthz
    python deploy.py --check    # verify and build only; no deploy
"""
import argparse
import ast
import filecmp
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile

REPO = os.path.dirname(os.path.abspath(__file__))
STAGING = os.path.join(os.environ.get("TEMP", "/tmp"), "lidar_deploy_staging")
ZIP_PATH = os.path.join(os.environ.get("TEMP", "/tmp"), "lidar_deploy.zip")

RESOURCE_GROUP = "LiDar_group-86b3"
APP_NAME = "LiDar"
SCM = "https://lidar-g4h0gnc8gngzftbw.scm.westus3-01.azurewebsites.net"
SITE = "https://lidar-g4h0gnc8gngzftbw.westus3-01.azurewebsites.net"

# Only these reach the server. Everything else in the repo is local scratch.
ROOT_FILES = ("app.py", "requirements.txt", "postbuild.sh")
PACKAGE_DIRS = ("gait-analysis", "motion-analysis")


def fail(msg):
    print(f"\nFAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def expected_motion_modules():
    """Read MOTION_MODULES out of app.py without importing it (no flask needed)."""
    tree = ast.parse(open(os.path.join(REPO, "app.py"), encoding="utf-8").read())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "MOTION_MODULES":
                    return list(ast.literal_eval(node.value))
    fail("MOTION_MODULES not found in app.py — the manifest guard needs it.")


def build_staging():
    if os.path.isdir(STAGING):
        shutil.rmtree(STAGING)
    os.makedirs(STAGING)

    for name in ROOT_FILES:
        src = os.path.join(REPO, name)
        if not os.path.isfile(src):
            fail(f"missing required file: {name}")
        shutil.copy2(src, os.path.join(STAGING, name))

    for pkg in PACKAGE_DIRS:
        src_dir = os.path.join(REPO, pkg)
        if not os.path.isdir(src_dir):
            fail(f"missing required directory: {pkg}")
        dst_dir = os.path.join(STAGING, pkg)
        os.makedirs(dst_dir)
        for fn in sorted(os.listdir(src_dir)):
            if fn.endswith(".py"):
                shutil.copy2(os.path.join(src_dir, fn), os.path.join(dst_dir, fn))
        print(f"  {pkg}: {len(os.listdir(dst_dir))} .py staged")


def verify_staging(expected):
    """Every expected module present, and byte-identical to the repo copy."""
    staged_dir = os.path.join(STAGING, "motion-analysis")
    staged = sorted(f[:-3] for f in os.listdir(staged_dir) if f.endswith(".py"))
    missing = [m for m in expected if m not in staged]
    if missing:
        fail(f"motion-analysis incomplete: {len(staged)}/{len(expected)} — missing {missing}")

    drifted = [m for m in expected
               if not filecmp.cmp(os.path.join(REPO, "motion-analysis", m + ".py"),
                                  os.path.join(staged_dir, m + ".py"), shallow=False)]
    if drifted:
        fail(f"staged copies differ from repo: {drifted}")
    print(f"  motion-analysis: {len(staged)}/{len(expected)} present, all byte-identical")


def build_zip():
    if os.path.exists(ZIP_PATH):
        os.remove(ZIP_PATH)
    n = 0
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        for dirpath, dirs, files in os.walk(STAGING):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for fn in sorted(files):
                full = os.path.join(dirpath, fn)
                # Forward slashes — Linux extraction mangles backslash entries.
                arc = os.path.relpath(full, STAGING).replace(os.sep, "/")
                zf.write(full, arc)
                n += 1
    print(f"  {n} files, {os.path.getsize(ZIP_PATH)/1e6:.2f} MB -> {ZIP_PATH}")


def verify_zip(expected):
    """Re-check the built artifact itself, not the staging dir it came from."""
    with zipfile.ZipFile(ZIP_PATH) as zf:
        names = zf.namelist()
    bad = [n for n in names if "\\" in n]
    if bad:
        fail(f"zip contains backslash paths (Linux will mangle these): {bad[:5]}")
    in_zip = sorted(n[len("motion-analysis/"):-3] for n in names
                    if n.startswith("motion-analysis/") and n.endswith(".py"))
    missing = [m for m in expected if m not in in_zip]
    if missing:
        fail(f"zip is missing motion-analysis modules: {missing}")
    print(f"  zip verified: {len(in_zip)}/{len(expected)} motion-analysis modules, no backslash paths")


def deploy():
    print("\n[4/5] Deploying (async; Oryx keeps building past the HTTP call)...")
    # --track-status defaults to True and makes the CLI wait for the Linux
    # worker to START, which is separate from --async and not what we want:
    # this app's build tarball is ~3 GB, so extraction routinely outruns the
    # CLI's 10-minute startup limit. The CLI then reports DEPLOYMENT FAILED
    # for a build that actually succeeded. We poll the deployment record and
    # gate on /healthz ourselves, so turn the CLI's own tracking off.
    r = subprocess.run(
        ["az", "webapp", "deploy", "--resource-group", RESOURCE_GROUP,
         "--name", APP_NAME, "--src-path", ZIP_PATH, "--type", "zip",
         "--async", "true", "--track-status", "false"],
        capture_output=True, text=True, shell=(os.name == "nt"))
    if r.returncode != 0:
        fail(f"az webapp deploy failed:\n{(r.stderr or r.stdout)[-1500:]}")

    token = subprocess.run(
        ["az", "account", "get-access-token", "--resource",
         "https://management.azure.com", "--query", "accessToken", "-o", "tsv"],
        capture_output=True, text=True, shell=(os.name == "nt")).stdout.strip()

    print("  polling build status...")
    for i in range(80):  # ~20 min
        time.sleep(15)
        try:
            req = urllib.request.Request(SCM + "/api/deployments/latest",
                                         headers={"Authorization": "Bearer " + token})
            with urllib.request.urlopen(req, timeout=60) as resp:
                d = json.loads(resp.read().decode())
        except Exception as e:
            print(f"  [{i*15:4}s] poll error: {e}")
            continue
        if d.get("complete"):
            # 4 = success, 3 = failed
            if d.get("status") == 4:
                print(f"  build succeeded at {d.get('end_time')}")
                restart()
                return
            fail(f"deployment failed (status {d.get('status')}): "
                 f"{d.get('status_text') or d.get('progress')}\n"
                 f"  If the SCM container restarted mid-build, just re-run this script.")
        print(f"  [{i*15:4}s] status={d.get('status')} {str(d.get('progress'))[:70]}")
    fail("deployment did not complete within 20 minutes")


def restart():
    """Oryx writes the new output.tar.zst, but the running container keeps
    serving the extraction it started with. Without this the deploy reports
    success while the site still runs the previous build.
    """
    print("  restarting container so the new build is actually served...")
    r = subprocess.run(["az", "webapp", "restart", "--resource-group",
                        RESOURCE_GROUP, "--name", APP_NAME],
                       capture_output=True, text=True, shell=(os.name == "nt"))
    if r.returncode != 0:
        fail(f"restart failed:\n{(r.stderr or r.stdout)[-800:]}")


def gate_on_healthz():
    print("\n[5/5] Gating on /healthz (200 = every module and profile imports)...")
    for i in range(40):  # container restart can take ~2 min
        time.sleep(15)
        try:
            with urllib.request.urlopen(SITE + "/healthz", timeout=120) as resp:
                return _report_health(json.loads(resp.read().decode()), resp.status)
        except urllib.error.HTTPError as e:
            body = e.read().decode()
            if e.code == 503:
                return _report_health(json.loads(body), 503)
            print(f"  [{i*15:4}s] HTTP {e.code}, still warming up")
        except Exception as e:
            print(f"  [{i*15:4}s] {type(e).__name__}, still warming up")
    fail("/healthz never responded")


def _report_health(d, code):
    # Staleness check first. The previous /healthz returned 200 with no "ok"
    # key and no per-file report, so "no failing checks" was indistinguishable
    # between a healthy new build and an old build still being served. Demand
    # the current schema before trusting anything else in the response.
    if "motion.files" not in d or "ok" not in d:
        fail("/healthz responded, but with the OLD schema — the container is "
             "still serving a previous build. The deploy did not take effect.")

    bad = {k: v for k, v in d.items()
           if isinstance(v, dict) and not v.get("ok")}
    files = d.get("motion.files", {})
    if files:
        print(f"  motion.files: {files.get('present')}/{files.get('expected')} present, "
              f"missing={files.get('missing')}")
    if d.get("ok") is not True:
        bad = bad or {"ok": {"error": "top-level ok is not True"}}
    if bad:
        for k, v in sorted(bad.items()):
            print(f"  FAIL {k} :: {v.get('error') or v}")
        fail(f"/healthz returned {code} — deploy is live but broken (see above)")
    print(f"  HTTP {code} — all checks green")
    print("\nDeploy verified. " + SITE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="verify and build the zip, but do not deploy")
    args = ap.parse_args()

    expected = expected_motion_modules()
    print(f"Manifest: {len(expected)} motion-analysis modules required\n")

    print("[1/5] Staging (allowlist)...")
    build_staging()
    print("[2/5] Verifying staging...")
    verify_staging(expected)
    print("[3/5] Building zip...")
    build_zip()
    verify_zip(expected)

    if args.check:
        print("\n--check: verified and built, not deployed.")
        return
    deploy()
    gate_on_healthz()


if __name__ == "__main__":
    main()
