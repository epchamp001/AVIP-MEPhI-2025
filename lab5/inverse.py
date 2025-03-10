from PIL import Image
import os

images_dir = 'images'
alphabet = 'АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ'


def invert_image(image_path, save_path):
    img = Image.open(image_path)

    img = img.convert('L')

    img = Image.eval(img, lambda x: 255 - x)

    img.save(save_path)


for symbol in alphabet:
    image_path = os.path.join(images_dir, f'{symbol}.png')
    if os.path.exists(image_path):
        save_path = os.path.join('inverse', f'{symbol}.png')
        invert_image(image_path, save_path)
