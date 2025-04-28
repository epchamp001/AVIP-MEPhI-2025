import os
from PIL import Image, ImageDraw, ImageFont

# Путь к файлу шрифта и папка вывода
FONT_PATH = 'fonts/TimesNewRoman.ttf'
OUTPUT_DIR = 'alphabet'

# Размер «рабочей области» для поиска оптимального размера шрифта
CANVAS_SIZE = (2048, 2048)  # больший холст, чтобы правильно вычислить bbox
BACKGROUND_COLOR = 'white'
TEXT_COLOR = 'black'

# Русские заглавные буквы
ALPHABET = list("АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_max_font_size(letter: str, canvas_size: tuple[int, int]) -> int:
    """
    Бинарным поиском находит максимальный размер шрифта,
    при котором буква полностью помещается в canvas_size.
    """
    min_size, max_size = 1, max(canvas_size)
    best = min_size
    img = Image.new('RGB', canvas_size)
    draw = ImageDraw.Draw(img)
    while min_size <= max_size:
        mid = (min_size + max_size) // 2
        font = ImageFont.truetype(FONT_PATH, mid)
        bbox = draw.textbbox((0, 0), letter, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if w <= canvas_size[0] and h <= canvas_size[1]:
            best = mid
            min_size = mid + 1
        else:
            max_size = mid - 1
    return best

for letter in ALPHABET:
    # 1. Найти оптимальный размер шрифта
    font_size = get_max_font_size(letter, CANVAS_SIZE)
    font = ImageFont.truetype(FONT_PATH, font_size)

    # 2. Отрисовать букву на большом холсте
    img = Image.new('RGB', CANVAS_SIZE, BACKGROUND_COLOR)
    draw = ImageDraw.Draw(img)
    # для точного центрирования в холсте
    bbox = draw.textbbox((0, 0), letter, font=font)
    x = (CANVAS_SIZE[0] - (bbox[2] - bbox[0])) / 2 - bbox[0]
    y = (CANVAS_SIZE[1] - (bbox[3] - bbox[1])) / 2 - bbox[1]
    draw.text((x, y), letter, font=font, fill=TEXT_COLOR)

    # 3. Создать маску текста и получить её bbox
    mask = Image.new('L', CANVAS_SIZE, 0)
    draw_mask = ImageDraw.Draw(mask)
    draw_mask.text((x, y), letter, font=font, fill=255)
    letter_bbox = mask.getbbox()  # (left, top, right, bottom)

    # 4. Добавить небольшой паддинг (например, 2 px), но не выходя за границы canvas
    pad = 2
    left   = max(letter_bbox[0] - pad, 0)
    top    = max(letter_bbox[1] - pad, 0)
    right  = min(letter_bbox[2] + pad, CANVAS_SIZE[0])
    bottom = min(letter_bbox[3] + pad, CANVAS_SIZE[1])

    # 5. Обрезать и сохранить
    cropped = img.crop((left, top, right, bottom))
    cropped.save(os.path.join(OUTPUT_DIR, f'{letter}.bmp'), format='BMP')

print("Генерация завершена. Файлы в папке:", OUTPUT_DIR)
