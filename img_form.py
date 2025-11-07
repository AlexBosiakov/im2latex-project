import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('C:/Users/Admin/Desktop/photo_2025-04-29_14-03-07.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite('results_gray.png', gray)

blur = cv2.GaussianBlur(gray, (5, 5), 0)
cv2.imwrite('results_blur.png', blur)

ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
cv2.imwrite('results_thresh.png', thresh)

plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1), plt.imshow(gray, cmap='gray'), plt.title('Gray')
plt.subplot(1, 3, 2), plt.imshow(blur, cmap='gray'), plt.title('Blur')
plt.subplot(1, 3, 3), plt.imshow(thresh, cmap='gray'), plt.title('Threshold')
plt.show()

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
img_contours = img.copy()
min_area = 100
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > min_area:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img_contours, (x, y), (x + w, y + h), (0, 255, 0), 2)
        segment = thresh[y:y + h, x:x + w]
        cv2.imwrite(f'results_segment_{x}_{y}.png', segment)
cv2.imwrite('results_segmented.png', img_contours)
plt.figure(figsize=(8, 6))
plt.imshow(cv2.cvtColor(img_contours, cv2.COLOR_BGR2RGB))
plt.title('Сегментированные регионы (символы)')
plt.axis('off')
plt.show()
