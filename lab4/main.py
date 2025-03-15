# import cv2
# import numpy as np
# import os
# from matplotlib import pyplot as plt
#
# G_x = np.array([[3, 10, 3],
#                 [0, 0, 0],
#                 [-3, -10, -3]], dtype=np.float32)
#
# G_y = np.array([[3, 0, -3],
#                 [10, 0, -10],
#                 [3, 0, -3]], dtype=np.float32)
#
# input_folder = 'pictures_src'
# output_folder = 'pictures_results'
#
# if not os.path.exists(output_folder):
#     os.makedirs(output_folder)
#
# image_files = [f for f in os.listdir(input_folder) if f.endswith('.png')]
#
# for image_file in image_files:
#     image_path = os.path.join(input_folder, image_file)
#
#     image = cv2.imread(image_path)
#
#     if image is None:
#         print(f"Ошибка: Не удалось загрузить изображение {image_file}. Пропускаем.")
#         continue
#
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#     # Применение оператора Шарра
#     gradient_x = cv2.filter2D(gray_image, -1, G_x)
#     gradient_y = cv2.filter2D(gray_image, -1, G_y)
#
#     gradient = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
#
#     if np.isnan(gradient).any() or np.isinf(gradient).any():
#         gradient = np.nan_to_num(gradient, nan=0.0, posinf=0.0, neginf=0.0)
#
#     gradient = gradient.astype(np.float32)
#
#     # Нормализация градиента к диапазону 0-255
#     gradient_normalized = cv2.normalize(gradient, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
#
#     # Бинаризация градиента
#     _, binary_gradient = cv2.threshold(gradient_normalized, 50, 255, cv2.THRESH_BINARY)
#
#     base_name = os.path.splitext(image_file)[0]
#     cv2.imwrite(os.path.join(output_folder, f'{base_name}_gray.png'), gray_image)
#     cv2.imwrite(os.path.join(output_folder, f'{base_name}_gradient_x.png'), gradient_x)
#     cv2.imwrite(os.path.join(output_folder, f'{base_name}_gradient_y.png'), gradient_y)
#     cv2.imwrite(os.path.join(output_folder, f'{base_name}_gradient_normalized.png'), gradient_normalized)
#     cv2.imwrite(os.path.join(output_folder, f'{base_name}_binary_gradient.png'), binary_gradient)
#
#     print(f"Обработано изображение: {image_file}")
#
# print("Обработка всех изображений завершена.")

import numpy as np
from PIL import Image
import os
import math

# Операторы Шарра
G_x = np.array([[3, 10, 3],
                [0, 0, 0],
                [-3, -10, -3]], dtype=np.float32)

G_y = np.array([[3, 0, -3],
                [10, 0, -10],
                [3, 0, -3]], dtype=np.float32)

# Функция для преобразования цветного изображения в полутоновое
def rgb_to_grayscale(image):
    return np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])

# Функция для применения свертки (реализация вручную)
def apply_convolution(image, kernel):
    kernel_height, kernel_width = kernel.shape
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2

    # Добавляем padding к изображению
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')

    # Создаем пустое изображение для результата
    output = np.zeros_like(image)

    # Применяем свертку
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            output[i, j] = np.sum(padded_image[i:i + kernel_height, j:j + kernel_width] * kernel)

    return output

# Функция для нормализации изображения к диапазону 0–255
def normalize_image(image):
    return (image - np.min(image)) / (np.max(image) - np.min(image)) * 255

# Функция для бинаризации изображения
def binarize_image(image, threshold):
    return (image > threshold) * 255

# Папки для входных и выходных данных
input_folder = 'pictures_src'
output_folder = 'pictures_results'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

image_files = [f for f in os.listdir(input_folder) if f.endswith('.png')]

# Обработка каждого изображения
for image_file in image_files:
    image_path = os.path.join(input_folder, image_file)

    # Загрузка изображения с помощью PIL
    image = np.array(Image.open(image_path))

    # Преобразование в полутоновое
    gray_image = rgb_to_grayscale(image)

    # Применение операторов Шарра
    gradient_x = apply_convolution(gray_image, G_x)
    gradient_y = apply_convolution(gray_image, G_y)

    # Вычисление градиента
    gradient = np.sqrt(gradient_x ** 2 + gradient_y ** 2)

    # Нормализация градиента
    gradient_normalized = normalize_image(gradient).astype(np.uint8)

    # Бинаризация градиента
    binary_gradient = binarize_image(gradient_normalized, threshold=50)

    # Сохранение результатов
    base_name = os.path.splitext(image_file)[0]
    Image.fromarray(gray_image.astype(np.uint8)).save(os.path.join(output_folder, f'{base_name}_gray.png'))
    Image.fromarray(gradient_x.astype(np.uint8)).save(os.path.join(output_folder, f'{base_name}_gradient_x.png'))
    Image.fromarray(gradient_y.astype(np.uint8)).save(os.path.join(output_folder, f'{base_name}_gradient_y.png'))
    Image.fromarray(gradient_normalized).save(os.path.join(output_folder, f'{base_name}_gradient_normalized.png'))
    Image.fromarray(binary_gradient.astype(np.uint8)).save(os.path.join(output_folder, f'{base_name}_binary_gradient.png'))

    print(f"Обработано изображение: {image_file}")

print("Обработка всех изображений завершена.")