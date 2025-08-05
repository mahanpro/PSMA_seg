"""
Usage:
    python anonymize_scans.py /path/to/folder  --csv out.csv  --dry-run
Drop --dry-run to perform the rename.
"""

import csv
import re
from pathlib import Path
import argparse

PATTERN = re.compile(r"^(?P<prefix>.+)_(?P<mod>(CT|PT))\.nii\.gz$", re.IGNORECASE)


def collect_files(folder: Path):
    """Return {prefix: {mod: file_path}}."""
    groups = {}
    for f in folder.glob("*.nii.gz"):
        m = PATTERN.match(f.name)
        if not m:
            continue  # skip non‑conforming files
        pfx, mod = m["prefix"], m["mod"].upper()
        groups.setdefault(pfx, {})[mod] = f
    return groups


def anonymize(folder: Path, csv_path: Path, digits=4, dry=True):
    groups = collect_files(folder)
    mapping_rows = []

    for idx, (pfx, scans) in enumerate(sorted(groups.items()), start=1):
        anon_id = str(idx).zfill(digits)  # 0001, 0002, …
        for mod, f in scans.items():
            new_name = f"{anon_id}_{mod}.nii.gz"
            target = f.with_name(new_name)
            if dry:
                print(f"[DRY] {f.name} -> {new_name}")
            else:
                f.rename(target)  # pathlib rename :contentReference[oaicite:0]{index=0}
        mapping_rows.append({"original": pfx, "anon_id": anon_id})

    # Write CSV
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["original", "anon_id"])
        writer.writeheader()
        writer.writerows(mapping_rows)
    print(f"Mapping saved to {csv_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=Path, help="Folder containing *.nii.gz")
    ap.add_argument("--csv", type=Path, default="mapping.csv")
    ap.add_argument("--dry-run", action="store_true", help="Preview without renaming")
    args = ap.parse_args()
    anonymize(args.folder, args.csv, dry=args.dry_run)
