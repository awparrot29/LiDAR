#!/bin/bash
# Runs after Oryx's `pip install -r requirements.txt` during the build phase.
#
# mediapipe depends on `opencv-contrib-python` — the full GUI build, which links
# against libGL/libxcb. Those system libraries are absent from the App Service
# Python image, and apt-get installs do NOT survive into the runtime container
# (only wwwroot, including the antenv venv, is carried over). Swapping in the
# headless build is therefore the only fix that persists.

set -u

PIP="pip"
for candidate in antenv/bin/pip /home/site/wwwroot/antenv/bin/pip; do
    if [ -x "$candidate" ]; then
        PIP="$candidate"
        break
    fi
done
echo "postbuild: using $PIP"

$PIP uninstall -y opencv-python opencv-contrib-python opencv-python-headless || true
$PIP install --no-cache-dir opencv-contrib-python-headless || exit 1

echo "postbuild: opencv swapped to headless build"
