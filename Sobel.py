import cv2
import numpy as np
import matplotlib.pyplot as plt

# Membaca gambar
img = cv2.imread("C:\\Users\\User\\Downloads\\ghos.jpg", 0)  # Membaca sebagai grayscale

# Fungsi konvolusi
def convolution(image, kernel):
    m, n = kernel.shape
    y, x = image.shape
    y = y - m + 1
    x = x - n + 1
    new_image = np.zeros((y, x))

    for i in range(y):
        for j in range(x):
            new_image[i][j] = (image[i:i+m, j:j+n]*kernel).sum()
    return new_image

# Kernel Robert
kernel_x = np.array([[0, 1], [-1, 0]])
kernel_y = np.array([[1, 0], [0, -1]])

# Kernel Sobel
kernel_x_sobel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
kernel_y_sobel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

# Hitung gradien untuk Robert
grad_x_robert = convolution(img, kernel_x)
grad_y_robert = convolution(img, kernel_y)
magnitude_robert = np.sqrt(grad_x_robert**2 + grad_y_robert**2)

# Hitung gradien untuk Sobel
grad_x_sobel = convolution(img, kernel_x_sobel)
grad_y_sobel = convolution(img, kernel_y_sobel)
magnitude_sobel = np.sqrt(grad_x_sobel**2 + grad_y_sobel**2)

# Visualisasi
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1), plt.imshow(img, cmap='gray'), plt.title('Gambar Asli')
plt.subplot(1, 3, 2), plt.imshow(magnitude_robert, cmap='gray'), plt.title('Deteksi Tepi Robert')
plt.subplot(1, 3, 3), plt.imshow(magnitude_sobel, cmap='gray'), plt.title('Deteksi Tepi Sobel')
plt.show()