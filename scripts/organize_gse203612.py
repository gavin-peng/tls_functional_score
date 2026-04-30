"""
Organize GSE203612 flat extracted files into per-sample Space Ranger directory structure.

Input (flat files in data/raw/GSE203612/):
  GSM6177618_NYU_PDAC1_Vis_processed_filtered_feature_bc_matrix.h5
  GSM6177618_NYU_PDAC1_Vis_processed_spatial_tissue_positions_list.csv.gz
  GSM6177618_NYU_PDAC1_Vis_processed_spatial_scalefactors_json.json.gz
  ...

Output (per-sample dirs):
  data/raw/GSE203612/NYU_PDAC1/
    filtered_feature_bc_matrix.h5
    spatial/
      tissue_positions_list.csv
      scalefactors_json.json

Usage:
  python scripts/organize_gse203612.py
"""

import gzip
import re
import shutil
from pathlib import Path

DATA_DIR = Path("data/raw/GSE203612")


def gunzip(src: Path, dst: Path):
    with gzip.open(src, 'rb') as fin, open(dst, 'wb') as fout:
        shutil.copyfileobj(fin, fout)


def main():
    flat_files = list(DATA_DIR.glob("GSM*_Vis_processed_*"))
    if not flat_files:
        print(f"No flat Vis_processed files found in {DATA_DIR}")
        print("Run the tar extraction first.")
        return

    print(f"Found {len(flat_files)} flat Visium files")

    # Parse: GSM6177618_NYU_PDAC1_Vis_processed_...
    pattern = re.compile(r"^(GSM\d+)_NYU_([A-Z0-9]+)_Vis_processed_(.*)")

    moved = 0
    for f in sorted(flat_files):
        m = pattern.match(f.name)
        if not m:
            print(f"  Skipping unrecognized: {f.name}")
            continue

        gsm_id, sample_short, rest = m.groups()
        sample_name = f"NYU_{sample_short}"

        # Determine target path
        if rest == "filtered_feature_bc_matrix.h5":
            dest_dir = DATA_DIR / sample_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / "filtered_feature_bc_matrix.h5"
            if dest.exists():
                print(f"  Already exists: {dest}")
            else:
                shutil.move(str(f), str(dest))
                print(f"  {f.name} -> {dest}")
                moved += 1

        elif rest.startswith("spatial_"):
            spatial_rest = rest[len("spatial_"):]   # e.g. tissue_positions_list.csv.gz
            dest_dir = DATA_DIR / sample_name / "spatial"
            dest_dir.mkdir(parents=True, exist_ok=True)

            if spatial_rest.endswith(".gz"):
                # Decompress
                base_name = spatial_rest[:-3]   # remove .gz
                dest = dest_dir / base_name
                if dest.exists():
                    print(f"  Already exists: {dest}")
                    f.unlink(missing_ok=True)
                else:
                    gunzip(f, dest)
                    f.unlink()
                    print(f"  {f.name} -> {dest} (decompressed)")
                    moved += 1
            else:
                dest = dest_dir / spatial_rest
                if dest.exists():
                    print(f"  Already exists: {dest}")
                else:
                    shutil.move(str(f), str(dest))
                    print(f"  {f.name} -> {dest}")
                    moved += 1
        else:
            print(f"  Unknown file type: {f.name}")

    print(f"\nMoved/organized {moved} files")

    # Report what we have
    print("\nSample directories:")
    for d in sorted(DATA_DIR.iterdir()):
        if d.is_dir() and d.name.startswith("NYU_"):
            h5 = d / "filtered_feature_bc_matrix.h5"
            pos = d / "spatial" / "tissue_positions_list.csv"
            sf  = d / "spatial" / "scalefactors_json.json"
            status = []
            if h5.exists():  status.append(f"h5({h5.stat().st_size//1024}KB)")
            if pos.exists(): status.append("positions")
            if sf.exists():  status.append("scalefactors")
            print(f"  {d.name}: {', '.join(status) if status else 'EMPTY'}")


if __name__ == "__main__":
    main()
