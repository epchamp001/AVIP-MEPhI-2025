import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

G_x = np.array([[3, 10, 3],
                [0, 0, 0],
                [-3, -10, -3]], dtype=np.float32)

G_y = np.array([[3, 0, -3],
                [10, 0, -10],
                [3, 0, -3]], dtype=np.float32)

input_folder = 'pictures_src'
output_folder = 'pictures_results'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

image_files = [f for f in os.listdir(input_folder) if f.endswith('.png')]

for image_file in image_files:
    image_path = os.path.join(input_folder, image_file)

    image = cv2.imread(image_path)

    if image is None:
        print(f"Ошибка: Не удалось загрузить изображение {image_file}. Пропускаем.")
        continue

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Применение оператора Шарра
    gradient_x = cv2.filter2D(gray_image, -1, G_x)
    gradient_y = cv2.filter2D(gray_image, -1, G_y)

    gradient = np.sqrt(gradient_x ** 2 + gradient_y ** 2)

    if np.isnan(gradient).any() or np.isinf(gradient).any():
        gradient = np.nan_to_num(gradient, nan=0.0, posinf=0.0, neginf=0.0)

    gradient = gradient.astype(np.float32)

    # Нормализация градиента к диапазону 0-255
    gradient_normalized = cv2.normalize(gradient, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Бинаризация градиента
    _, binary_gradient = cv2.threshold(gradient_normalized, 50, 255, cv2.THRESH_BINARY)

    base_name = os.path.splitext(image_file)[0]
    cv2.imwrite(os.path.join(output_folder, f'{base_name}_gray.png'), gray_image)
    cv2.imwrite(os.path.join(output_folder, f'{base_name}_gradient_x.png'), gradient_x)
    cv2.imwrite(os.path.join(output_folder, f'{base_name}_gradient_y.png'), gradient_y)
    cv2.imwrite(os.path.join(output_folder, f'{base_name}_gradient_normalized.png'), gradient_normalized)
    cv2.imwrite(os.path.join(output_folder, f'{base_name}_binary_gradient.png'), binary_gradient)

    print(f"Обработано изображение: {image_file}")

print("Обработка всех изображений завершена.")