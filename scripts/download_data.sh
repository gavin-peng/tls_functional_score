#!/usr/bin/env bash
# Download public datasets for TLS functional state scoring project.
#
# Usage:
#   bash scripts/download_data.sh all          # download everything
#   bash scripts/download_data.sh rcc          # GSE175540 only
#   bash scripts/download_data.sh multicancer  # GSE203612 only

set -euo pipefail

DATA_DIR="data/raw"
mkdir -p "$DATA_DIR"

# ── Helpers ──────────────────────────────────────────────────────────────────

check_tool() {
    command -v "$1" >/dev/null 2>&1 || { echo "ERROR: $1 not found. Install it first."; exit 1; }
}

geo_download() {
    # Download a GEO series using GEOparse or wget
    local accession="$1"
    local dest="$DATA_DIR/$accession"
    mkdir -p "$dest"
    echo "→ Downloading $accession to $dest"
    # Use GEOparse Python API (handles FTP automatically)
    python - <<EOF
import GEOparse, os
gse = GEOparse.get_GEO(geo="$accession", destdir="$dest", silent=False)
print(f"Downloaded {len(gse.gsms)} samples")
EOF
}

# ── Dataset: GSE175540 — RCC Visium (Immunity 2022) ───────────────────────────
download_rcc() {
    echo ""
    echo "=== GSE175540: RCC Visium (TLS + ICB outcomes) ==="
    local dest="$DATA_DIR/GSE175540"
    mkdir -p "$dest"

    # Supplementary files: Space Ranger outputs per sample
    echo "Downloading via GEO FTP..."
    wget -q -r -nH --cut-dirs=5 --no-parent \
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE175nnn/GSE175540/suppl/" \
        -P "$dest" \
        --accept "*.tar.gz,*.h5,*.csv,*.tsv,*.gz" \
        --reject "index.html*" || true

    # Also try using GEOparse
    python - <<EOF
import GEOparse
import os
dest = "$dest"
os.makedirs(dest, exist_ok=True)
try:
    gse = GEOparse.get_GEO(geo="GSE175540", destdir=dest, silent=False)
    print(f"Found {len(gse.gsms)} samples in GSE175540")
    for gsm_name, gsm in gse.gsms.items():
        print(f"  {gsm_name}: {gsm.metadata.get('title', [''])[0]}")
except Exception as e:
    print(f"GEOparse error: {e}")
    print("Try manual download from: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE175540")
EOF
    echo "GSE175540 download complete → $dest"
}

# ── Dataset: GSE203612 — Multi-cancer Visium ─────────────────────────────────
download_multicancer() {
    echo ""
    echo "=== GSE203612: Multi-cancer Visium (9 cancer types) ==="
    local dest="$DATA_DIR/GSE203612"
    mkdir -p "$dest"

    python - <<EOF
import GEOparse
import os
dest = "$dest"
os.makedirs(dest, exist_ok=True)
try:
    gse = GEOparse.get_GEO(geo="GSE203612", destdir=dest, silent=False)
    print(f"Found {len(gse.gsms)} samples")
except Exception as e:
    print(f"GEOparse error: {e}")
    print("Manual: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE203612")
EOF
    echo "GSE203612 download attempt complete → $dest"
}

# ── Main ──────────────────────────────────────────────────────────────────────
check_tool python
check_tool wget
# Install GEOparse if missing
python -c "import GEOparse" 2>/dev/null || pip install -q GEOparse

TARGET="${1:-all}"
case "$TARGET" in
    rcc)         download_rcc ;;
    multicancer) download_multicancer ;;
    all)
        download_rcc
        download_multicancer
        ;;
    *)
        echo "Unknown target: $TARGET"
        echo "Usage: bash scripts/download_data.sh [all|rcc|multicancer]"
        exit 1
        ;;
esac

echo ""
echo "All downloads complete. Data in: $DATA_DIR"
echo "Next step: python scripts/preprocess_all.py"
