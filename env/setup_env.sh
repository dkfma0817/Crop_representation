#!/usr/bin/env bash
# Rebuild the experiment env (Python 3.10.19, torch 2.7.1+cu118) from env/requirements.txt.
#   bash env/setup_env.sh [ENV_PATH]      default: /workspace/Areum/research/envs/dinov2_ord_py310
# Put ENV_PATH on persistent storage (not a container-local conda dir).
set -euo pipefail

ENV="${1:-/workspace/Areum/research/envs/dinov2_ord_py310}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Ignore ~/.local site-packages: otherwise pip treats packages there as "already satisfied"
# and skips installing them into the env (seen with sympy/safetensors/antlr4).
export PYTHONNOUSERSITE=1

if ! command -v conda >/dev/null 2>&1; then
  if [ ! -x "$HOME/miniforge3/bin/conda" ]; then
    echo "[setup] conda not found -> installing miniforge to $HOME/miniforge3"
    wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh
    bash /tmp/miniforge.sh -b -p "$HOME/miniforge3"
  fi
  export PATH="$HOME/miniforge3/bin:$PATH"
fi

if [ -x "$ENV/bin/python" ]; then
  echo "[setup] $ENV already exists -> reusing it (delete it first for a clean rebuild)"
else
  conda create -p "$ENV" python=3.10.19 -y
fi
conda env config vars set PYTHONNOUSERSITE=1 -p "$ENV" >/dev/null

"$ENV/bin/pip" install -r "$HERE/requirements.txt" --extra-index-url https://download.pytorch.org/whl/cu118

"$ENV/bin/python" - <<'EOF'
import torch, timm, transformers, lpips  # noqa: F401
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(), "| cudnn", torch.backends.cudnn.version(),
      "| timm", timm.__version__, "| transformers", transformers.__version__)
print("expected: torch 2.7.1+cu118 | cuda True | cudnn 90100 | timm 1.0.27 | transformers 4.52.1")
EOF
echo "[setup] done: $ENV/bin/python"
