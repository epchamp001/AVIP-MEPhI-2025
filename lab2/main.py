import os
import numpy as np
import cv2

input_folder = 'pictures_src/'
output_folder = 'pictures_results/'

def load_image(image_name):
    return cv2.imread(os.path.join(input_folder, image_name))

def save_image(image, image_name):
    output_path = os.path.join(output_folder, image_name)
    cv2.imwrite(output_path, image)

# Приведение изображения к полутоновому
def to_grayscale(image):
    grayscale = 0.299 * image[:, :, 2] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 0]
    return grayscale.astype(np.uint8)

# Бинаризация изображения
def binarize_image(grayscale_image, threshold=128):
    binary_image = grayscale_image > threshold
    return binary_image.astype(np.uint8) * 255

# Адаптивная бинаризация Сингха (Окно 5x5)
def adaptive_binarization(image, window_size=5):
    grayscale_image = to_grayscale(image)
    output_image = np.zeros_like(grayscale_image)
    padding = window_size // 2
    for i in range(padding, grayscale_image.shape[0] - padding):
        for j in range(padding, grayscale_image.shape[1] - padding):
            window = grayscale_image[i - padding:i + padding + 1, j - padding:j + padding + 1]
            threshold = np.mean(window)
            output_image[i, j] = 255 if grayscale_image[i, j] > threshold else 0
    return output_image

image_name = 'text.png'

image = load_image(image_name)

grayscale_image = to_grayscale(image)
save_image(grayscale_image, 'grayscale_' + image_name)

binary_image = binarize_image(grayscale_image, threshold=108)
save_image(binary_image, 'binary_' + image_name)

adaptive_binary_image = adaptive_binarization(image, window_size=5)
save_image(adaptive_binary_image, 'adaptive_binary_' + image_name)
