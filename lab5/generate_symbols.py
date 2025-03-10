from PIL import Image, ImageDraw, ImageFont
import os

output_folder = "images"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

letters = "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"

font = ImageFont.truetype("Times New Roman.ttf", 52)

for letter in letters:
    img = Image.new('L', (100, 100), color=255)  # цвет 255 (белый)
    draw = ImageDraw.Draw(img)

    text_width, text_height = draw.textbbox((0, 0), letter, font=font)[2:4]  # Новый способ получения размеров
    position = ((100 - text_width) // 2, (100 - text_height) // 2)

    draw.text(position, letter, fill=0, font=font)

    img.save(f"{output_folder}/{letter}.png")
