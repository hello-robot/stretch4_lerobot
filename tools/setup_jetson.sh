#!/usr/bin/env bash
# Create the Jetson (JetPack 6) venv for stretch4_policy_server: CUDA PyTorch for Python 3.10,
# LeRobot 0.6.1 patched to 3.10 syntax, and this repository. LeRobot 0.6 needs Python >= 3.12,
# but NVIDIA's CUDA PyTorch for JetPack 6 exists only for 3.10 (PyPI's aarch64 wheels have no
# CUDA for the Orin). After reinstalling lerobot by hand, rerun this script (or the patch).
#
#   tools/setup_jetson.sh                 # venv at <PATH>/stretch4_lerobot/.venv (this repository)
#   tools/setup_jetson.sh /path/to/venv   # or somewhere else
#
# Safe to rerun: it reuses the venv and reinstalls into it.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${1:-$REPO/.venv}"
PYTHON="${PYTHON:-python3}"

if [ "$(id -u)" -eq 0 ]; then
    echo "Run this without sudo: the venv must belong to your user." >&2
    exit 1
fi
if [ "$("$PYTHON" -c 'import sys; print("%d.%d" % sys.version_info[:2])')" != "3.10" ]; then
    echo "$PYTHON is not Python 3.10 (JetPack 6's CUDA PyTorch is 3.10 only); set PYTHON=..." >&2
    exit 1
fi
no_venv() {
    echo "$PYTHON cannot create a venv. Install it, then rerun this script:" >&2
    echo "    sudo apt update && sudo apt install python3.10-venv" >&2
    exit 1
}
"$PYTHON" -c "import ensurepip" 2>/dev/null || no_venv

# A venv left without bin/activate by a failed run is rebuilt (--clear).
if [ ! -f "$VENV/bin/activate" ]; then
    if [ -e "$VENV" ] && [ ! -f "$VENV/pyvenv.cfg" ]; then
        echo "$VENV exists and is not a venv; remove it or pass another path." >&2
        exit 1
    fi
    "$PYTHON" -m venv --clear "$VENV" || no_venv
fi
activate() { set +u; source "$VENV/bin/activate"; set -u; }
activate
pip install --upgrade pip

# PyTorch for JetPack 6. 2.9.1, not 2.8: the 2.8 build is compiled against NumPy 1.x and LeRobot
# needs NumPy 2. --no-deps and only this index, or pip takes PyPI's CPU-only build.
pip install --no-cache-dir --no-deps --index-url https://pypi.jetson-ai-lab.io/jp6/cu126/+simple \
    "torch==2.9.1" "torchvision==0.24.1"
pip install "numpy>=2,<2.3" sympy networkx jinja2 fsspec filelock typing-extensions pillow

# torch 2.9.1 links cuDSS, which JetPack 6 does not ship
pip install --no-cache-dir nvidia-cudss-cu12
pip uninstall -y nvidia-cublas-cu12 nvidia-cuda-nvrtc-cu12 cuda-toolkit   # keep the system CUDA's
if ! grep -q "# cudss_lib" "$VENV/bin/activate"; then
    echo 'export LD_LIBRARY_PATH=$VIRTUAL_ENV/lib/python3.10/site-packages/nvidia/cu12/lib:${LD_LIBRARY_PATH:-}  # cudss_lib' \
        >> "$VENV/bin/activate"
fi
activate

# LeRobot 0.6.1 on Python 3.10, pinned to that torch
printf "torch==2.9.1\ntorchvision==0.24.1\nnumpy<2.3\n" > "$VENV/constraints.txt"
pip install --no-cache-dir --only-binary=:all: "matplotlib>=3.10.3,<4" "contourpy>=1.3,<2"
pip install --no-cache-dir --ignore-requires-python -c "$VENV/constraints.txt" \
    "lerobot[async,diffusion]==0.6.1"
python "$REPO/tools/patch_lerobot_py310.py" \
    "$(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")/lerobot"

# this repository (its packages say Python >= 3.12; the patched LeRobot runs them on 3.10).
pip install --no-deps --ignore-requires-python \
    -e "$REPO/packages/lerobot_camera_oak" -e "$REPO/packages/lerobot_robot_stretch4"

python -c "import torch, numpy, lerobot_robot_stretch4.policy_server; \
print('torch', torch.__version__, 'numpy', numpy.__version__, 'cuda', torch.cuda.is_available())"
echo "Done. Run:  source $VENV/bin/activate && stretch4_policy_server --host=0.0.0.0 --port=8080"
