from PIL import Image
import os

root_dir = "data/processed"

bad_files = []

for root, dirs, files in os.walk(root_dir):
    for file in files:
        if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")):
            path = os.path.join(root, file)
            try:
                with Image.open(path) as img:
                    img.verify()
            except Exception as e:
                bad_files.append((path, str(e)))

print("Bad files found:", len(bad_files))
for path, err in bad_files:
    print(path, "->", err)