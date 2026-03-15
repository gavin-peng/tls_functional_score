#!/usr/bin/env bash
# Install PyTorch + torch-geometric ecosystem in the correct order.
#
# Must be run AFTER: conda env create -f environment.yml && conda activate tls_spatial
#
# Usage:
#   bash scripts/install_torch.sh          # auto-detects CPU vs CUDA
#   bash scripts/install_torch.sh cpu      # force CPU-only
#   bash scripts/install_torch.sh cu121    # force CUDA 12.1

set -euo pipefail

# ── Detect platform ───────────────────────────────────────────────────────────
if [[ "${1:-auto}" == "cpu" ]]; then
    PLATFORM="cpu"
elif [[ "${1:-auto}" == cu* ]]; then
    PLATFORM="$1"
else
    if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
        # Parse CUDA version from nvidia-smi
        CUDA_VER=$(nvidia-smi | grep -oP 'CUDA Version: \K[\d.]+' | head -1)
        MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
        MINOR=$(echo "$CUDA_VER" | cut -d. -f2)
        PLATFORM="cu${MAJOR}${MINOR}"
        echo "Detected GPU with CUDA ${CUDA_VER} → using platform: $PLATFORM"
    else
        PLATFORM="cpu"
        echo "No GPU detected → installing CPU-only PyTorch"
    fi
fi

echo "Platform: $PLATFORM"
echo ""

# ── Step 1: Install PyTorch ───────────────────────────────────────────────────
echo "=== Step 1: Installing PyTorch ==="
if [[ "$PLATFORM" == "cpu" ]]; then
    pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu
else
    pip install torch==2.2.2 --index-url "https://download.pytorch.org/whl/${PLATFORM}"
fi

# Verify
python -c "import torch; print(f'PyTorch {torch.__version__} installed. CUDA: {torch.cuda.is_available()}')"

# ── Step 2: Get installed torch version for wheel URL ─────────────────────────
TORCH_VER=$(python -c "import torch; v=torch.__version__; print(v.split('+')[0])")
echo ""
echo "=== Step 2: Installing torch-geometric ==="
pip install torch-geometric==2.5.3

# ── Step 3: Install torch-scatter and torch-sparse via pre-built wheels ───────
echo ""
echo "=== Step 3: Installing torch-scatter, torch-sparse, torch-cluster ==="
WHEEL_URL="https://data.pyg.org/whl/torch-${TORCH_VER}+${PLATFORM}.html"
echo "Wheel index: $WHEEL_URL"
pip install torch-scatter torch-sparse torch-cluster -f "$WHEEL_URL"

# ── Step 4: Verify full install ───────────────────────────────────────────────
echo ""
echo "=== Verification ==="
python - <<'EOF'
import torch
print(f"  torch:          {torch.__version__}")
import torch_geometric
print(f"  torch-geometric: {torch_geometric.__version__}")
try:
    import torch_scatter
    print(f"  torch-scatter:  OK")
except ImportError as e:
    print(f"  torch-scatter:  FAILED — {e}")
try:
    import torch_sparse
    print(f"  torch-sparse:   OK")
except ImportError as e:
    print(f"  torch-sparse:   FAILED — {e}")
try:
    import torch_cluster
    print(f"  torch-cluster:  OK")
except ImportError as e:
    print(f"  torch-cluster:  FAILED — {e}")

# Quick GNN sanity check
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
x = torch.randn(10, 8)
edge_index = torch.randint(0, 10, (2, 20))
conv = GATConv(8, 16, heads=2)
out = conv(x, edge_index)
print(f"\nGATConv forward pass OK: {x.shape} -> {out.shape}")
print("\nAll checks passed!")
EOF

echo ""
echo "Done. Activate with: conda activate tls_spatial"
