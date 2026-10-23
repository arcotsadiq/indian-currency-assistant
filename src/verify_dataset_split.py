from pathlib import Path
from collections import Counter
import hashlib

root = Path("data/currency_split")
train_dir = root / "images/train"
val_dir = root / "images/val"

def image_hashes(folder):
    return {
        hashlib.sha256(p.read_bytes()).hexdigest()
        for p in folder.iterdir()
        if p.is_file()
    }

train_hashes = image_hashes(train_dir)
val_hashes = image_hashes(val_dir)

print("Training images:", len(list(train_dir.iterdir())))
print("Validation images:", len(list(val_dir.iterdir())))
print("Identical images shared between sets:", len(train_hashes & val_hashes))

for split in ("train", "val"):
    label_dir = root / "labels" / split
    counts = Counter()
    for label in label_dir.glob("*.txt"):
        for line in label.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if parts:
                counts[int(parts[0])] += 1

    print(f"\n{split.upper()} class counts:")
    for class_id, name in enumerate(
        ["10 INR", "20 INR", "50 INR", "100 INR", "200 INR", "500 INR"]
    ):
        print(f"{name}: {counts[class_id]}")
