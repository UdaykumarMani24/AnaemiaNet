"""
00_prepare_cpanemic_dataset.py
──────────────────────────────
Prepares the CP-AnemiC dataset (your local folder) for the AnemiCare pipeline.

Your dataset structure:
    CP-AnemiC dataset/
        Anemia_Data_Collection_Sheet.xlsx   ← metadata (IMAGE_ID, HB_LEVEL, REMARK …)
        Anemic/                             ← 424 images
        Non-anemic/                         ← 286 images

Output structure (ready for 02_train_model.py):
    data/cpanemic_processed/
        train/  anemic/  non_anemic/
        val/    anemic/  non_anemic/
        test/   anemic/  non_anemic/
        metadata.csv                        ← full sheet + split assignment

Usage
─────
    python 00_prepare_cpanemic_dataset.py \
        --dataset_dir "C:/Users/SASTRA/Desktop/aneapp/CP-AnemiC dataset"

    # Optional: also export severity sub-splits (Mild / Moderate / Severe)
    python 00_prepare_cpanemic_dataset.py \
        --dataset_dir "C:/Users/SASTRA/Desktop/aneapp/CP-AnemiC dataset" \
        --severity_splits
"""

import argparse
import shutil
import random
from pathlib import Path

import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
EXCEL_FILENAME  = "Anemia_Data_Collection_Sheet.xlsx"
ANEMIC_FOLDER   = "Anemic"
NON_ANEMIC_FOLDER = "Non-anemic"

TRAIN_SPLIT = 0.75
VAL_SPLIT   = 0.15
TEST_SPLIT  = 0.10
SEED        = 42

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# Matches your Excel REMARK column values
LABEL_MAP = {
    "anemic":     "anemic",
    "non-anemic": "non_anemic",
    "non_anemic": "non_anemic",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def find_image(image_id: str, search_dirs: list[Path]) -> Path | None:
    """
    Locate an image file by IMAGE_ID across multiple source folders.
    Tries exact name + common extensions.
    """
    for folder in search_dirs:
        for ext in IMG_EXTENSIONS:
            candidate = folder / f"{image_id}{ext}"
            if candidate.exists():
                return candidate
            # Some datasets use lowercase extension
            candidate = folder / f"{image_id}{ext.lower()}"
            if candidate.exists():
                return candidate
    return None


def split_indices(n: int) -> tuple[range, range, range]:
    """Return (train, val, test) index ranges for n items."""
    n_train = int(n * TRAIN_SPLIT)
    n_val   = int(n * VAL_SPLIT)
    return (
        range(0, n_train),
        range(n_train, n_train + n_val),
        range(n_train + n_val, n),
    )


def copy_images(rows, src_dirs, out_dir, label, split_name, split_col_updates):
    """Copy images for one (label, split) group; record split in metadata."""
    dest = out_dir / split_name / label
    dest.mkdir(parents=True, exist_ok=True)
    copied, missing = 0, 0

    for _, row in rows:
        img_path = find_image(row["IMAGE_ID"], src_dirs)
        if img_path is None:
            print(f"  [!] Missing image: {row['IMAGE_ID']}")
            missing += 1
            split_col_updates[row["IMAGE_ID"]] = f"MISSING_{split_name}"
            continue
        dst = dest / f"{row['IMAGE_ID']}{img_path.suffix}"
        shutil.copy2(img_path, dst)
        copied += 1
        split_col_updates[row["IMAGE_ID"]] = split_name

    return copied, missing


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Prepare CP-AnemiC dataset for AnemiCare training pipeline"
    )
    parser.add_argument(
        "--dataset_dir", type=Path, required=True,
        help='Path to "CP-AnemiC dataset" folder containing the Excel + image folders'
    )
    parser.add_argument(
        "--out_dir", type=Path, default=Path("data/cpanemic_processed"),
        help="Output directory (default: data/cpanemic_processed)"
    )
    parser.add_argument(
        "--severity_splits", action="store_true",
        help="Also create severity-stratified sub-folders (Mild/Moderate/Severe)"
    )
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    out_dir     = args.out_dir

    # ── 1. Load Excel metadata ────────────────────────────────────────────────
    excel_path = dataset_dir / EXCEL_FILENAME
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel not found: {excel_path}")

    df = pd.read_excel(excel_path)
    df.columns = df.columns.str.strip()
    print(f"[✓] Loaded metadata: {len(df)} rows")

    # Normalise REMARK → binary label
    df["label"] = (
        df["REMARK"]
        .str.strip()
        .str.lower()
        .map(LABEL_MAP)
    )
    unmapped = df["label"].isna().sum()
    if unmapped:
        print(f"  [!] {unmapped} rows with unrecognised REMARK values — skipped")
        df = df.dropna(subset=["label"])

    print(f"     anemic:     {(df['label']=='anemic').sum()}")
    print(f"     non_anemic: {(df['label']=='non_anemic').sum()}")

    # ── 2. Source image directories ───────────────────────────────────────────
    anemic_dir     = dataset_dir / ANEMIC_FOLDER
    non_anemic_dir = dataset_dir / NON_ANEMIC_FOLDER

    for d in (anemic_dir, non_anemic_dir):
        if not d.exists():
            raise FileNotFoundError(
                f"Image folder not found: {d}\n"
                f"Expected inside: {dataset_dir}"
            )

    src_dirs = [anemic_dir, non_anemic_dir]

    # ── 3. Split per class (stratified) ──────────────────────────────────────
    random.seed(SEED)
    split_assignments: dict[str, str] = {}   # IMAGE_ID → split name
    stats: dict[str, dict[str, int]] = {}

    for label in ("anemic", "non_anemic"):
        label_rows = list(df[df["label"] == label].iterrows())
        random.shuffle(label_rows)
        n = len(label_rows)

        tr_idx, va_idx, te_idx = split_indices(n)
        splits = {
            "train": [label_rows[i] for i in tr_idx],
            "val":   [label_rows[i] for i in va_idx],
            "test":  [label_rows[i] for i in te_idx],
        }

        stats[label] = {}
        print(f"\n[→] Copying {label} …")
        for split_name, rows in splits.items():
            copied, missing = copy_images(
                rows, src_dirs, out_dir, label, split_name, split_assignments
            )
            stats[label][split_name] = copied
            if missing:
                print(f"     {split_name}: {copied} copied, {missing} MISSING")
            else:
                print(f"     {split_name}: {copied} images")

    # ── 4. Optional severity splits (anemic sub-classes) ─────────────────────
    if args.severity_splits:
        print("\n[→] Creating severity sub-splits (Mild / Moderate / Severe) …")
        sev_out = out_dir / "severity_splits"
        for severity in ("Mild", "Moderate", "Severe"):
            sev_rows = list(df[df["Severity"] == severity].iterrows())
            random.shuffle(sev_rows)
            n = len(sev_rows)
            if n == 0:
                continue
            tr_idx, va_idx, te_idx = split_indices(n)
            for split_name, idx_range in (
                ("train", tr_idx), ("val", va_idx), ("test", te_idx)
            ):
                rows = [sev_rows[i] for i in idx_range]
                dest = sev_out / split_name / severity.lower()
                dest.mkdir(parents=True, exist_ok=True)
                for _, row in rows:
                    img_path = find_image(row["IMAGE_ID"], src_dirs)
                    if img_path:
                        shutil.copy2(img_path, dest / f"{row['IMAGE_ID']}{img_path.suffix}")
            print(f"  {severity}: {n} images → train/val/test")

    # ── 5. Save enriched metadata CSV ────────────────────────────────────────
    df["split"] = df["IMAGE_ID"].map(split_assignments)
    metadata_path = out_dir / "metadata.csv"
    df.to_csv(metadata_path, index=False)
    print(f"\n[✓] Metadata saved → {metadata_path}")

    # ── 6. Print summary ──────────────────────────────────────────────────────
    print("\n" + "═" * 56)
    print("  CP-AnemiC dataset preparation complete")
    print("═" * 56)
    for split in ("train", "val", "test"):
        total = sum(v.get(split, 0) for v in stats.values())
        print(f"\n  {split.upper():5s}  ({total} images)")
        for label, counts in stats.items():
            print(f"    {label:12s} {counts.get(split, 0):>4d}")

    print(f"\n  Output → {out_dir.resolve()}")
    print("═" * 56)
    print("\nNext step:")
    print(f"  python 02_train_model.py --data_dir {out_dir}")
    print()


if __name__ == "__main__":
    main()
