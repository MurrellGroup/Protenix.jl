#!/usr/bin/env bash
set -euo pipefail

# One-shot setup for Python PXDesign reference inference on CPU.
# Installs into a workspace-local venv so no global site-packages writes are needed.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/opt/homebrew/bin/python3.11}"
VENV_DIR="${VENV_DIR:-$ROOT/.venv_pyref}"
EXTERNAL_DIR="${EXTERNAL_DIR:-$ROOT/.external}"
PXDESIGN_PY_ROOT="${PXDESIGN_PY_ROOT:-$(cd "$ROOT/.." && pwd)}"
PROTENIX_DIR="$EXTERNAL_DIR/Protenix"
PXDBENCH_DIR="$EXTERNAL_DIR/PXDesignBench"
CCD_DIR="$ROOT/release_data/ccd_cache"
CKPT_DIR="$ROOT/release_data/checkpoint"

echo "[setup] ROOT=$ROOT"
echo "[setup] PYTHON_BIN=$PYTHON_BIN"
echo "[setup] VENV_DIR=$VENV_DIR"
echo "[setup] PXDESIGN_PY_ROOT=$PXDESIGN_PY_ROOT"

command -v "$PYTHON_BIN" >/dev/null 2>&1 || {
  echo "[error] python binary not found: $PYTHON_BIN"
  exit 1
}

mkdir -p "$EXTERNAL_DIR" "$CCD_DIR" "$CKPT_DIR"

if [ ! -d "$PROTENIX_DIR/.git" ]; then
  echo "[setup] cloning Protenix..."
  git clone --depth 1 --branch v0.5.0+pxd https://github.com/bytedance/Protenix.git "$PROTENIX_DIR"
else
  echo "[setup] Protenix already present, ensuring tag v0.5.0+pxd..."
  git -C "$PROTENIX_DIR" fetch --tags --depth 1 origin v0.5.0+pxd || true
  git -C "$PROTENIX_DIR" checkout v0.5.0+pxd
fi

if [ ! -d "$PXDBENCH_DIR/.git" ]; then
  echo "[setup] cloning PXDesignBench..."
  git clone --depth 1 https://github.com/bytedance/PXDesignBench.git "$PXDBENCH_DIR"
else
  echo "[setup] PXDesignBench already present."
fi

if [ ! -d "$VENV_DIR" ]; then
  echo "[setup] creating venv..."
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

echo "[setup] installing Python deps..."
python -m pip install --upgrade pip setuptools wheel
python -m pip install --upgrade uv || true
UV_BIN="$VENV_DIR/bin/uv"
if [ -x "$UV_BIN" ]; then
  echo "[setup] installing deps with uv..."
  "$UV_BIN" pip install --python "$VENV_DIR/bin/python" \
    numpy==1.26.4 \
    pandas==2.2.3 \
    PyYAML \
    scipy \
    ml_collections \
    tqdm \
    optree \
    biopython==1.83 \
    modelcif==0.7 \
    biotite==1.0.1 \
    scikit-learn \
    scikit-learn-extra \
    protobuf==3.20.2 \
    posix_ipc \
    einops \
    natsort \
    dm-tree \
    requests \
    networkx \
    msgpack \
    rdkit==2024.3.3
else
  echo "[setup] uv unavailable, falling back to pip..."
  python -m pip install \
    numpy==1.26.4 \
    pandas==2.2.3 \
    PyYAML \
    scipy \
    ml_collections \
    tqdm \
    optree \
    biopython==1.83 \
    modelcif==0.7 \
    biotite==1.0.1 \
    scikit-learn \
    scikit-learn-extra \
    protobuf==3.20.2 \
    posix_ipc \
    einops \
    natsort \
    dm-tree \
    requests \
    networkx \
    msgpack \
    rdkit==2024.3.3
fi

echo "[setup] installing local editable packages..."
if [ -x "$UV_BIN" ]; then
  "$UV_BIN" pip install --python "$VENV_DIR/bin/python" --no-deps -e "$PROTENIX_DIR"
  "$UV_BIN" pip install --python "$VENV_DIR/bin/python" --no-deps -e "$PXDBENCH_DIR"
  "$UV_BIN" pip install --python "$VENV_DIR/bin/python" --no-deps -e "$PXDESIGN_PY_ROOT"
else
  python -m pip install --no-deps -e "$PROTENIX_DIR"
  python -m pip install --no-deps -e "$PXDBENCH_DIR"
  python -m pip install --no-deps -e "$PXDESIGN_PY_ROOT"
fi

if [ ! -f "$CCD_DIR/components.v20240608.cif" ]; then
  echo "[setup] downloading CCD components..."
  curl -fL -o "$CCD_DIR/components.v20240608.cif" \
    https://pxdesign.tos-cn-beijing.volces.com/release_data/components.v20240608.cif
fi
if [ ! -f "$CCD_DIR/components.v20240608.cif.rdkit_mol.pkl" ]; then
  echo "[setup] downloading CCD RDKit cache..."
  curl -fL -o "$CCD_DIR/components.v20240608.cif.rdkit_mol.pkl" \
    https://pxdesign.tos-cn-beijing.volces.com/release_data/components.v20240608.cif.rdkit_mol.pkl
fi
if [ ! -f "$CCD_DIR/clusters-by-entity-40.txt" ]; then
  echo "[setup] downloading cluster file..."
  curl -fL -o "$CCD_DIR/clusters-by-entity-40.txt" \
    https://pxdesign.tos-cn-beijing.volces.com/release_data/clusters-by-entity-40.txt
fi

# PXDesign's inference helper attempts to download these if missing.
for f in protenix_base_default_v0.5.0.pt protenix_mini_default_v0.5.0.pt protenix_mini_tmpl_v0.5.0.pt; do
  if [ ! -f "$CKPT_DIR/$f" ]; then
    echo "[setup] creating placeholder checkpoint to skip optional tool download: $f"
    : > "$CKPT_DIR/$f"
  fi
done

cat > "$ROOT/scripts/python_reference_env.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
ROOT="$ROOT"
source "$VENV_DIR/bin/activate"
export PROTENIX_DATA_ROOT_DIR="$CCD_DIR"
export PYTHONPATH="$PXDESIGN_PY_ROOT:$PROTENIX_DIR:$PXDBENCH_DIR"
EOF
chmod +x "$ROOT/scripts/python_reference_env.sh"

echo "[setup] done."
echo "[setup] activate via: source $ROOT/scripts/python_reference_env.sh"
