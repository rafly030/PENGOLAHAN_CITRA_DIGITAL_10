import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt

# Fungsi untuk melakukan deteksi tepi dengan operator Robert
def roberts_operator(image):
    # Kernel untuk operator Robert
    kernel_x = np.array([[1, 0], [0, -1]])
    kernel_y = np.array([[0, 1], [-1, 0]])

    # Konvolusi dengan kernel Robert
    grad_x = convolve(image, kernel_x)
    grad_y = convolve(image, kernel_y)

    # Magnitudo gradien
    edge_roberts = np.sqrt(grad_x**2 + grad_y**2)
    return edge_roberts

# Fungsi untuk melakukan deteksi tepi dengan operator Sobel
def sobel_operator(image):
    # Kernel untuk operator Sobel
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    # Konvolusi dengan kernel Sobel
    grad_x = convolve(image, kernel_x)
    grad_y = convolve(image, kernel_y)

    # Magnitudo gradien
    edge_sobel = np.sqrt(grad_x**2 + grad_y**2)
    return edge_sobel

# Fungsi untuk melakukan konvolusi
def convolve(image, kernel):
    kernel_height, kernel_width = kernel.shape
    image_height, image_width = image.shape

    # Padding untuk mempertahankan ukuran gambar
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant', constant_values=0)

    # Hasil konvolusi
    result = np.zeros_like(image)

    # Operasi konvolusi
    for i in range(image_height):
        for j in range(image_width):
            region = padded_image[i:i+kernel_height, j:j+kernel_width]
            result[i, j] = np.sum(region * kernel)

    return result

# Memuat gambar grayscale
def load_image(image_path):
    image = imageio.imread(image_path, mode='F')  # Memperbaiki parameter mode
    return image / 255.0  # Normalisasi ke rentang 0-1

# Main program
if __name__ == "__main__":
    # Path gambar
    image_path = "C:\\Users\\User\\Downloads\\cod.jpg"

    # Memuat gambar
    image = load_image(image_path)

    # Deteksi tepi menggunakan operator Robert
    edge_roberts = roberts_operator(image)

    # Deteksi tepi menggunakan operator Sobel
    edge_sobel = sobel_operator(image)

    # Visualisasi hasil
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(image, cmap='gray')
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Robert Operator")
    plt.imshow(edge_roberts, cmap='gray')
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Sobel Operator")
    plt.imshow(edge_sobel, cmap='gray')
    plt.axis("off")

    plt.tight_layout()
    plt.show()

# Analisa:
# 1. Operator Robert lebih sederhana dan cocok untuk mendeteksi tepi dengan detail kecil, namun hasilnya lebih kasar dibanding Sobel.
# 2. Operator Sobel menghasilkan tepi yang lebih halus karena kernel yang lebih besar (3x3), sehingga lebih tahan terhadap noise.