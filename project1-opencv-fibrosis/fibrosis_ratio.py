import cv2
import numpy as np

# 1. Load H&E kidney slide (put a sample.jpg next to this file)
img = cv2.imread("sample.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 2. Convert to HSV for better color separation
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# 3. Simple threshold for blue fibrotic areas (Masson-like)
# tune these values for your own slides
lower_blue = np.array([90, 30, 30])
upper_blue = np.array([130, 255, 255])
mask = cv2.inRange(hsv, lower_blue, upper_blue)

# 4. Calculate fibrosis ratio
fibrosis_pixels = np.count_nonzero(mask)
total_pixels = mask.size
ratio = fibrosis_pixels / total_pixels * 100

print(f"Fibrosis ratio: {ratio:.2f}%")

# 5. Save result
cv2.imwrite("fibrosis_mask.jpg", mask)
print("Mask saved as fibrosis_mask.jpg")
