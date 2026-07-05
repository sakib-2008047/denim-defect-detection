import os

folder_path = "dataset/resized"
prefix = "pinhole"  # change per defect class

files = sorted(os.listdir(folder_path))
count = 1

for file in files:
    if not file.lower().endswith(('.jpg', '.png', '.jpeg')):
        continue

    old_path = os.path.join(folder_path, file)
    new_name = f"{prefix}_{count:04d}.jpg"
    new_path = os.path.join(folder_path, new_name)

    os.rename(old_path, new_path)
    count += 1

print("Renaming completed successfully!")
