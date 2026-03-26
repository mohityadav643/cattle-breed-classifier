import os
import shutil
import random

base_path = "dataset/all"
output_base = "dataset"

split_ratio = (0.7, 0.2, 0.1)

print("Folders:", os.listdir(base_path))

for folder in ["train", "val", "test"]:
    path = os.path.join(output_base, folder)
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

for breed in os.listdir(base_path):
    breed_path = os.path.join(base_path, breed)
    images = os.listdir(breed_path)

    random.shuffle(images)

    total = len(images)
    train_end = int(split_ratio[0] * total)
    val_end = int((split_ratio[0] + split_ratio[1]) * total)

    splits = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:]
    }

    for split in splits:
        split_dir = os.path.join(output_base, split, breed)
        os.makedirs(split_dir, exist_ok=True)

        for img in splits[split]:
            shutil.copy(
                os.path.join(breed_path, img),
                os.path.join(split_dir, img)
            )

print("✅ SPLIT DONE")