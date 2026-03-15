#!/usr/bin/env bash
# Download Visium samples from GSE203612 (multi-cancer, 9 types)
# Only downloads spatial Visium samples (not scRNA-seq libs)
#
# Usage: bash scripts/download_gse203612_visium.sh
# Output: data/raw/GSE203612/{SAMPLE_NAME}/ in Space Ranger format

set -euo pipefail

BASE_FTP="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE203nnn/GSE203612/suppl"
OUT_DIR="data/raw/GSE203612"
mkdir -p "$OUT_DIR"

# Visium samples only (those with _Vis_processed_ files + spatial coords)
# Format: "GSM_ID|sample_name"
SAMPLES=(
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
    local url="$1"
    local dest="$2"
    if [[ -f "$dest" ]]; then
        echo "  Already exists: $dest"
    else
        echo "  Downloading: $(basename $dest)"
        wget -q --show-progress -O "$dest" "$url" || {
            echo "  FAILED: $url"
            rm -f "$dest"
        }
    fi
}

for entry in "${SAMPLES[@]}"; do
    GSM="${entry%%|*}"
    NAME="${entry##*|}"
    PREFIX="${GSM}_NYU_${NAME##NYU_}_Vis_processed"
    # PDX entries don't have NYU prefix; handled separately if needed
    SDIR="$OUT_DIR/$NAME"
    mkdir -p "$SDIR/spatial"

    echo ""
    echo "=== $NAME ($GSM) ==="

    # Count matrix (h5)
    H5_URL="${BASE_FTP}/${PREFIX}_filtered_feature_bc_matrix.h5"
    dl "$H5_URL" "$SDIR/filtered_feature_bc_matrix.h5"

    # Spatial files
    for SPATIAL_FILE in \
        "tissue_positions_list.csv" \
        "scalefactors_json.json" \
        "tissue_hires_image.png" \
        "tissue_lowres_image.png"; do

        EXT="${SPATIAL_FILE##*.}"
        if [[ "$EXT" == "csv" || "$EXT" == "json" ]]; then
            # These are stored gzipped on GEO
            GZ_NAME="${PREFIX}_spatial_${SPATIAL_FILE}.gz"
            DEST_GZ="$SDIR/spatial/${SPATIAL_FILE}.gz"
            DEST="$SDIR/spatial/${SPATIAL_FILE}"
            if [[ -f "$DEST" ]]; then
                echo "  Already exists: $DEST"
            else
                dl "${BASE_FTP}/${GZ_NAME}" "$DEST_GZ"
                [[ -f "$DEST_GZ" ]] && gunzip -f "$DEST_GZ" || true
            fi
        else
            # Images stored gzipped
            GZ_NAME="${PREFIX}_spatial_${SPATIAL_FILE}.gz"
            DEST_GZ="$SDIR/spatial/${SPATIAL_FILE}.gz"
            DEST="$SDIR/spatial/${SPATIAL_FILE}"
            if [[ -f "$DEST" ]]; then
                echo "  Already exists: $DEST"
            else
                dl "${BASE_FTP}/${GZ_NAME}" "$DEST_GZ"
                [[ -f "$DEST_GZ" ]] && gunzip -f "$DEST_GZ" || true
            fi
        fi
    done

    echo "  Done: $SDIR"
done

echo ""
echo "All Visium samples downloaded to: $OUT_DIR"
echo "Next: python scripts/build_multicancer_h5ad.py"
