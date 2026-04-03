import os

for split in ["train", "val", "test"]:
    path = f"data/processed/{split}"
    print(f"\nChecking {path}")
    for item in os.listdir(path):
        print("-", item)