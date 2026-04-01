from pathlib import Path
import random
import shutil
import subprocess
import sys

RANDOM_SEED = 42
MIN_IMAGES_PER_CLASS = 180
MAX_IMAGES_PER_CLASS = 380
MAX_TOTAL_IMAGES = 14000
MIN_TOTAL_IMAGES = 4500
REQUIRED_CLASSES = [
    "Apple___Cedar_apple_rust",
    "Apple___Apple_scab",
]
DATASET_REPO = "https://github.com/spMohanty/PlantVillage-Dataset.git"
RAW_DATASET_ROOT = Path("data/raw_plantvillage")

source = RAW_DATASET_ROOT / "raw" / "color"
out_train = Path("data/train")
out_val = Path("data/validation")
out_test = Path("data/test")


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png"}


def ensure_source_dataset() -> None:
    """Clone PlantVillage source dataset if it is not already present."""
    if source.exists():
        return

    RAW_DATASET_ROOT.parent.mkdir(parents=True, exist_ok=True)
    print("Source dataset not found. Cloning PlantVillage repository...")
    result = subprocess.run(
        ["git", "clone", "--depth", "1", DATASET_REPO, str(RAW_DATASET_ROOT)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError("Failed to clone PlantVillage dataset source.")

    if not source.exists():
        raise RuntimeError(f"Dataset clone completed but expected path is missing: {source}")


def reset_split_dirs() -> None:
    for split_dir in [out_train, out_val, out_test]:
        split_dir.mkdir(parents=True, exist_ok=True)
        for child in split_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child)


def choose_classes():
    classes = []
    for class_dir in sorted(source.iterdir()):
        if not class_dir.is_dir():
            continue
        images = [p for p in class_dir.iterdir() if is_image_file(p)]
        if len(images) >= MIN_IMAGES_PER_CLASS:
            classes.append((class_dir.name, len(images)))

    classes = sorted(classes, key=lambda x: x[1], reverse=True)
    available = {name: count for name, count in classes}
    selected = []

    # Include mandatory classes first when available.
    for class_name in REQUIRED_CLASSES:
        if class_name not in available:
            continue
        selected.append((class_name, available[class_name]))

    # Then include all remaining eligible classes for maximum coverage.
    for class_name, count in classes:
        if any(class_name == chosen for chosen, _ in selected):
            continue
        selected.append((class_name, count))

    return selected


def resolve_per_class_target(selected_classes: list[tuple[str, int]]) -> int:
    if not selected_classes:
        raise RuntimeError("No classes selected for split preparation")

    min_class_count = min(count for _, count in selected_classes)
    per_class_by_total = max(1, MAX_TOTAL_IMAGES // len(selected_classes))
    return min(MAX_IMAGES_PER_CLASS, min_class_count, per_class_by_total)


def split_and_copy(class_name: str, per_class_target: int) -> tuple[int, int, int]:
    class_source = source / class_name
    images = [p for p in class_source.iterdir() if is_image_file(p)]
    random.shuffle(images)
    images = images[:per_class_target]

    total = len(images)
    n_train = int(0.7 * total)
    n_val = int(0.15 * total)

    train_images = images[:n_train]
    val_images = images[n_train:n_train + n_val]
    test_images = images[n_train + n_val:]

    split_map = {
        out_train: train_images,
        out_val: val_images,
        out_test: test_images,
    }

    for split_dir, split_images in split_map.items():
        class_target = split_dir / class_name
        class_target.mkdir(parents=True, exist_ok=True)
        split_prefix = "train" if split_dir == out_train else "val" if split_dir == out_val else "test"
        for idx, image_path in enumerate(split_images):
            # Keep destination paths short on Windows to avoid MAX_PATH copy failures.
            new_name = f"{split_prefix}_{idx:04d}{image_path.suffix.lower()}"
            shutil.copy2(image_path, class_target / new_name)

    return len(train_images), len(val_images), len(test_images)


def main() -> None:
    ensure_source_dataset()

    random.seed(RANDOM_SEED)
    reset_split_dirs()

    selected_classes = choose_classes()
    if len(selected_classes) < 8:
        raise RuntimeError(
            f"Only found {len(selected_classes)} classes with >= {MIN_IMAGES_PER_CLASS} images"
        )

    per_class_target = resolve_per_class_target(selected_classes)
    if per_class_target < MIN_IMAGES_PER_CLASS:
        raise RuntimeError(
            f"Per-class target too small ({per_class_target}). Lower MIN_IMAGES_PER_CLASS or MAX_TOTAL_IMAGES constraints."
        )

    selected_names = {name for name, _ in selected_classes}
    missing_required = [c for c in REQUIRED_CLASSES if c not in selected_names]
    if missing_required:
        print("WARNING: Missing required classes in prepared splits:")
        for class_name in missing_required:
            print(f"  - {class_name}")

    print("Selected classes:")
    for class_name, count in selected_classes:
        print(f"  {class_name}: {count}")
    print(f"\nClasses selected: {len(selected_classes)}")
    print(f"Per-class sample target: {per_class_target}")
    print(f"Expected total: {per_class_target * len(selected_classes)}")

    selected_plants = sorted({c.split("___")[0] for c, _ in selected_classes})
    print(f"\nUnique plants represented ({len(selected_plants)}): {', '.join(selected_plants)}")

    total_train = total_val = total_test = 0
    for class_name, _ in selected_classes:
        n_train, n_val, n_test = split_and_copy(class_name, per_class_target)
        total_train += n_train
        total_val += n_val
        total_test += n_test
        print(f"{class_name} -> train={n_train}, validation={n_val}, test={n_test}")

    total = total_train + total_val + total_test
    print("\nSummary:")
    print(f"  Train: {total_train}")
    print(f"  Validation: {total_val}")
    print(f"  Test: {total_test}")
    print(f"  Total: {total}")

    if total < MIN_TOTAL_IMAGES:
        print(f"WARNING: Total sample count is below target minimum ({MIN_TOTAL_IMAGES}).")
        sys.exit(2)

    if total > MAX_TOTAL_IMAGES:
        print(f"WARNING: Total sample count is above max target ({MAX_TOTAL_IMAGES}).")
        sys.exit(2)


if __name__ == "__main__":
    main()
