from pathlib import Path
import random
import shutil

source = Path(r"C:\Users\arcot\Downloads\Indian-Currency-DataSet-main\Indian-Currency-DataSet-main")
destination = Path("data/currency_split")

random.seed(42)

images = sorted((source / "images/train").glob("*"))
images = [p for p in images if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}]

pairs = []
for image in images:
    label = source / "labels/train" / f"{image.stem}.txt"
    if not label.exists():
        raise FileNotFoundError(f"Missing label: {label}")
    pairs.append((image, label))

random.shuffle(pairs)
split_index = int(len(pairs) * 0.8)
train_pairs = pairs[:split_index]
val_pairs = pairs[split_index:]

for split, split_pairs in [("train", train_pairs), ("val", val_pairs)]:
    for image, label in split_pairs:
        image_out = destination / "images" / split / image.name
        label_out = destination / "labels" / split / label.name
        image_out.parent.mkdir(parents=True, exist_ok=True)
        label_out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image, image_out)
        shutil.copy2(label, label_out)

yaml_path = destination / "currency.yaml"
yaml_path.write_text(
    f"path: {destination.resolve().as_posix()}\n"
    "train: images/train\n"
    "val: images/val\n\n"
    "names:\n"
    '  0: "10 INR"\n'
    '  1: "20 INR"\n'
    '  2: "50 INR"\n'
    '  3: "100 INR"\n'
    '  4: "200 INR"\n'
    '  5: "500 INR"\n',
    encoding="utf-8"
)

print("New dataset created:", destination.resolve())
print("Training pairs:", len(train_pairs))
print("Validation pairs:", len(val_pairs))
print("Dataset YAML:", yaml_path.resolve())
