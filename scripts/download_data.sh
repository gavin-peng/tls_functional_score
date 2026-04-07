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
    local base_ftp="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE203nnn/GSE203612/suppl"

    # Visium-only samples: GSM_ID|sample_name
    local samples=(
        "GSM6177599|NYU_BRCA0"
        "GSM6177601|NYU_BRCA1"
        "GSM6177603|NYU_BRCA2"
        "GSM6177607|NYU_GIST1"
        "GSM6177609|NYU_GIST2"
        "GSM6177612|NYU_LIHC1"
        "GSM6177614|NYU_OVCA1"
        "GSM6177617|NYU_OVCA3"
        "GSM6177618|NYU_PDAC1"
        "GSM6177623|NYU_UCEC3"
    )

    dl() {
        local url="$1" out="$2"
        if [[ -f "$out" ]]; then
            echo "  Already exists: $out"
        else
            echo "  Downloading: $(basename "$out")"
            wget -q --show-progress -O "$out" "$url" || { echo "  FAILED: $url"; rm -f "$out"; }
        fi
    }

    for entry in "${samples[@]}"; do
        local gsm="${entry%%|*}"
        local name="${entry##*|}"
        local prefix="${gsm}_NYU_${name##NYU_}_Vis_processed"
        local sdir="$dest/$name"
        mkdir -p "$sdir/spatial"

        echo ""
        echo "=== $name ($gsm) ==="

        dl "${base_ftp}/${prefix}_filtered_feature_bc_matrix.h5" \
           "$sdir/filtered_feature_bc_matrix.h5"

        for f in tissue_positions_list.csv scalefactors_json.json \
                  tissue_hires_image.png tissue_lowres_image.png; do
            local gz="$sdir/spatial/${f}.gz"
            local out="$sdir/spatial/${f}"
            if [[ -f "$out" ]]; then
                echo "  Already exists: $out"
            else
                dl "${base_ftp}/${prefix}_spatial_${f}.gz" "$gz"
                [[ -f "$gz" ]] && gunzip -f "$gz" || true
            fi
        done

        echo "  Done: $sdir"
    done
    echo "GSE203612 download complete → $dest"
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
