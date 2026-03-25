from pathlib import Path
import random 
import shutil

def split_dataset(
        raw_data_dir: str = "data/raw",
        processed_data_dir: str = "data/processed",
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
) -> None:
    if round(train_ratio + val_ratio + test_ratio, 2) != 1.0:
        raise ValueError("Train, validation, and test ratios must add up to 1.0")

    random.seed(seed)

    raw_path = Path(raw_data_dir)
    processed_path = Path(processed_data_dir)

    train_path = processed_path / "train"
    val_path = processed_path / "val"
    test_path = processed_path / "test"

    for split_path in [train_path, val_path, test_path]:
        split_path.mkdir(parents=True, exist_ok=True)

    class_dirs = [d for d in raw_path.iterdir() if d.is_dir()]

    if not class_dirs:
        raise FileNotFoundError(f"No class folders found in {raw_path}")
        
    for class_dir in class_dirs:
            images = [f for f in class_dir.iterdir() if f.is_file()]
            random.shuffle(images)

            total_images = len(images)
            train_end = int(total_images * train_ratio)
            val_end = train_end + int(total_images * val_ratio)

            train_files = images[:train_end]
            val_files = images[train_end:val_end]
            test_files = images[val_end:]

            for split_name, split_files in {
                "train": train_files,
                "val": val_files,
                "test": test_files,
            }.items():
                target_class_dir = processed_path / split_name / class_dir.name
                target_class_dir.mkdir(parents=True, exist_ok=True)

                for file_path in split_files:
                    shutil.copy2(file_path, target_class_dir / file_path.name)

            print(
                f"{class_dir.name}: "
                f"{len(train_files)} train,"
                f"{len(val_files)} val,"
                f"{len(test_files)} test"
            )
    print("Dataset split complete.")

    if __name__ == "__main__":
        split_dataset()
        
        
