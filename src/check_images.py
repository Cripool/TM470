from pathlib import Path
from PIL import Image

data_dir = Path("data/processed")
bad_files = []

for image_path in data_dir.rglob("*"):
    if image_path.is_file() and image_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
        try:
            with Image.open(image_path) as img:
                img.load()
        except Exception as e:
            bad_files.append((image_path, str(e)))

print(f"Found {len(bad_files)} bad files")

for path, err in bad_files:
    print(path)
    print(err)