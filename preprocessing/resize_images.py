import cv2
import os

input_folder = "dataset/original"
output_folder = "dataset/resized"
os.makedirs(output_folder, exist_ok=True)

IMG_SIZE = 224  # target resolution (pixels)

for img_name in os.listdir(input_folder):
    img_path = os.path.join(input_folder, img_name)
    img = cv2.imread(img_path)
    if img is None:
        continue

    resized_img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    save_path = os.path.join(output_folder, img_name)
    cv2.imwrite(save_path, resized_img)

print("All images resized successfully!")
