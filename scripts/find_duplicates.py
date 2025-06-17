#!/usr/bin/env python3
"""
Checks for duplicate images between training and validation sets
and optionally removes or moves them from the validation set.
Uses perceptual hash distance to detect near-duplicates.
"""

import shutil
from pathlib import Path
from PIL import Image
import imagehash
from tqdm import tqdm

# === SETTINGS ===
REMOVE_DUPLICATES = False           # True --> Delete duplicates; False --> Move duplicates
QUARANTINE_FOLDER = "unusedData/Duplicates"
MAX_DISTANCE = 5                   # Max hamming distance for considering images duplicates

def get_image_hashes(folder):
    """Returns dict {path: hash} for all images in folder"""
    hashes = {}
    for img_path in tqdm(list(folder.rglob("*.[pj][np]g")), desc=f"Hashing {folder.name}"):
        try:
            with Image.open(img_path) as img:
                h = imagehash.phash(img)
                hashes[img_path] = h
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
    return hashes

def handle_duplicates(duplicates, remove=True, quarantine_dir="data/Duplicates"):
    """Deletes or moves duplicates from the validation set"""
    if not duplicates:
        print("✅ No duplicates found.")
        return

    if remove:
        print("\n🧹 Removing duplicates from validation set...")
        for _, val_path in duplicates:
            try:
                val_path.unlink()
                print(f"Deleted: {val_path}")
            except Exception as e:
                print(f"❌ Failed to delete {val_path}: {e}")
    else:
        print(f"\n📁 Moving duplicates to '{quarantine_dir}'...")
        quarantine_dir = Path(quarantine_dir) / "Validation"
        quarantine_dir.mkdir(parents=True, exist_ok=True)
        for _, val_path in duplicates:
            try:
                dest = quarantine_dir / val_path.name
                shutil.move(str(val_path), dest)
                print(f"Moved: {val_path} → {dest}")
            except Exception as e:
                print(f"❌ Failed to move {val_path}: {e}")

def main():
    train_dir = Path("data/Training")
    val_dir = Path("data/Testing")

    print("🔍 Scanning for image duplicates between training and validation sets...\n")

    train_hashes = get_image_hashes(train_dir)  # {path: hash}
    val_hashes = get_image_hashes(val_dir)      # {path: hash}

    duplicates = []
    for val_path, val_hash in tqdm(val_hashes.items(), desc="Comparing validation images"):
        for train_path, train_hash in train_hashes.items():
            distance = val_hash - train_hash
            if distance <= MAX_DISTANCE:
                duplicates.append((train_path, val_path))
                # Once duplicate found for this val image, no need to check more
                break

    print(f"\n⚠️ Found {len(duplicates)} duplicate images between training and validation sets.")

    if duplicates:
        print("\n📝 Example duplicates:")
        for train_path, val_path in duplicates[:10]:
            print(f"Train: {train_path} | Val: {val_path}")

        print("\n⚠️ Data leakage detected! Taking action...\n")
        handle_duplicates(duplicates, remove=REMOVE_DUPLICATES, quarantine_dir=QUARANTINE_FOLDER)
    else:
        print("No duplicates found. All good!")

if __name__ == "__main__":
    main()
