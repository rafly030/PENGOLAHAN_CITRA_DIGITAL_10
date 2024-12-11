import cv2
import numpy as np
import matplotlib.pyplot as plt

# Membaca gambar
img = cv2.imread("C:\\Users\\User\\Downloads\\ghos.jpg", 0)

# Deteksi tepi menggunakan Canny
edges = cv2.Canny(img, 100, 200)

# Visualisasi
plt.subplot(1, 2, 1),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(1, 2, 2),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()