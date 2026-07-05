import cv2
import os
import numpy as np

input_folder = r"F:\final image\Original\Oil Strain"
output_folder = r"F:\final image\Augmented\Oil Strain"
os.makedirs(output_folder, exist_ok=True)


def adjust_brightness(image, value):
    """Shift the V channel of an HSV image by 'value' (positive = brighter)."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = np.clip(v.astype(int) + value, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)


count = 1

for file in os.listdir(input_folder):
    if not file.lower().endswith(('.jpg', '.png', '.jpeg')):
        continue

    img = cv2.imread(os.path.join(input_folder, file))
    if img is None:
        continue

    def save(image):
        global count
        cv2.imwrite(
            os.path.join(output_folder, f"Oil_Strain_{count:04d}.jpg"),
            image
        )
        count += 1

    save(img)                              # original
    save(cv2.rotate(img, cv2.ROTATE_180))  # 180 degree rotation
    save(cv2.flip(img, 1))                 # horizontal flip
    save(cv2.flip(img, 0))                 # vertical flip
    save(adjust_brightness(img, 20))       # brighter
    save(adjust_brightness(img, -20))      # darker

print("Data augmentation completed successfully!")
