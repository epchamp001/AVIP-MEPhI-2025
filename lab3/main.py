import cv2
import numpy as np
import os

input_folder = 'pictures_src'
output_folder = 'pictures_results'

os.makedirs(output_folder, exist_ok=True)

def process_image(image_path, output_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print(f"Не удалось загрузить изображение {image_path}.")
        return

    kernel = np.array([[0, 1, 0],
                       [1, 1, 1],
                       [0, 1, 0]], dtype=np.uint8)  # Кольцо

    opened_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    diff_image = cv2.absdiff(image, opened_image)

    filtered_path = os.path.join(output_folder, f"filtered_{os.path.basename(image_path)}")
    diff_path = os.path.join(output_folder, f"diff_{os.path.basename(image_path)}")

    cv2.imwrite(filtered_path, opened_image)
    cv2.imwrite(diff_path, diff_image)

    print(f"Изображение {os.path.basename(image_path)} обработано и сохранено в {output_folder}.")

for image_name in os.listdir(input_folder):
    image_path = os.path.join(input_folder, image_name)
    if os.path.isfile(image_path):
        process_image(image_path, output_folder)
