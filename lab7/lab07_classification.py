import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

"""
Лабораторная работа № 7 — Классификация по признакам (упрощённый подход)
Вариант 21: русский алфавит (заглавные буквы)

Используем готовую сегментацию из ЛР‑6: берём каждую вырезанную букву
со снимка фразы, сравниваем её непосредственно с эталонным изображением
буквы из каталога `alphabet/` и выбираем то, что совпадает лучше всего.

Метрика сходства — сумма абсолютных различий (SAD) между нормализованными
бинарными изображениями размером 64×64. Чем меньше SAD, тем выше сходство.

Скрипт запускается одной командой:
    python lab7_feature_classification.py

Выводит в консоль распознанную строку и статистику точности.
"""

###############################################################################
# Константы каталога
###############################################################################

SRC_PATH = Path("../lab6/pictures_src/phrase.bmp")          # исходное изображение
ALPHABET_DIR = Path("alphabet")                     # 33 файлов *.bmp «А».bmp, …
DST_DIR = Path("pictures_results")                  # для отладочных артефактов
PHRASE_GT = "БЕРЕГИТЕ В СЕБЕ ЧЕЛОВЕКА"              # эталонная фраза
SIZE = (64, 64)                                      # единый размер шаблонов

###############################################################################
# Утилиты работы с изображениями
###############################################################################

def ensure_dirs():
    DST_DIR.mkdir(parents=True, exist_ok=True)


def to_binary(path: Path | Image.Image) -> np.ndarray:
    """Конвертирует BMP‑файл или объект PIL в бинарную матрицу 0/1."""
    img = Image.open(path) if isinstance(path, (str, Path)) else path
    img = img.convert("1")  # 1‑битный (чёрно‑белый) режим
    arr = 1 - np.array(img, dtype=np.uint8)  # 1 — чёрный пиксель
    return arr


def resize_bin(arr: np.ndarray, size=SIZE) -> np.ndarray:
    """Масштабирует бинарное изображение до size с сохранением 0/1."""
    pil = Image.fromarray((1 - arr) * 255)  # обратно в картинку, белый‑чёрный
    pil = pil.resize(size, Image.NEAREST)
    return 1 - np.array(pil, dtype=np.uint8)

###############################################################################
# Сегментация символов (код из ЛР‑6, слегка вынесен в функции)
###############################################################################

THRESHOLD = 1  # «пустой» столбец/строка <= 1 чёрный пиксель


def segment_by_profiles(bin_img: np.ndarray, threshold: int = THRESHOLD):
    h, w = bin_img.shape
    vert = bin_img.sum(axis=0)

    splits = []
    in_char = False
    x_start = 0
    for x, v in enumerate(vert):
        if not in_char and v > threshold:
            in_char = True
            x_start = x
        elif in_char and v <= threshold:
            in_char = False
            splits.append((x_start, x - 1))
    if in_char:
        splits.append((x_start, w - 1))

    boxes = []
    for x0, x1 in splits:
        char_slice = bin_img[:, x0:x1 + 1]
        horiz = char_slice.sum(axis=1)
        ys = np.where(horiz > threshold)[0]
        if ys.size == 0:
            continue
        y0, y1 = ys[0], ys[-1]
        boxes.append((x0, y0, x1, y1))
    return boxes


def gap_is_space(prev_box, curr_box, space_ratio=0.5):
    """Эвристика пробела — промежуток > space_ratio * средняя ширина буквы."""
    if prev_box is None:
        return False
    prev_right = prev_box[2]
    curr_left = curr_box[0]
    prev_w = prev_box[2] - prev_box[0]
    return (curr_left - prev_right) > prev_w * space_ratio

###############################################################################
# Загрузка эталонных букв
###############################################################################

def load_templates() -> dict[str, np.ndarray]:
    templates = {}
    for p in sorted(ALPHABET_DIR.glob("*.bmp")):
        ch = p.stem.upper()
        bin_img = to_binary(p)
        templates[ch] = resize_bin(bin_img)
    if len(templates) != 33:
        print("[!] Предупреждение: ожидалось 33 шаблона, найдено", len(templates))
    return templates

###############################################################################
# Основная логика распознавания
###############################################################################

def recognise_image(img_path: Path, templates: dict[str, np.ndarray]):
    bin_img = to_binary(img_path)
    boxes = segment_by_profiles(bin_img)
    chars = []
    result_str = ""
    prev_box = None

    # предрассчёт для шаблонов — стек в ndarray для скорости
    tpl_stack = np.stack(list(templates.values()))  # shape: (33, 64, 64)
    tpl_keys = list(templates.keys())

    for box in boxes:
        x0, y0, x1, y1 = box
        sub = bin_img[y0:y1 + 1, x0:x1 + 1]
        sub = resize_bin(sub)
        # расширяем до (1,64,64) и вычисляем |sub - template|, суммируем
        diff = np.abs(tpl_stack - sub)
        sad = diff.sum(axis=(1, 2))
        best_idx = int(sad.argmin())
        best_char = tpl_keys[best_idx]
        chars.append(best_char)

        if gap_is_space(prev_box, box):
            result_str += " "
        result_str += best_char
        prev_box = box

    return result_str, boxes, chars

###############################################################################
# Метрика точности
###############################################################################

def accuracy(pred: str, gt: str):
    m = max(len(pred), len(gt))
    errs = sum(1 for a, b in zip(pred.ljust(m), gt.ljust(m)) if a != b)
    return errs, 100 * (1 - errs / m)

###############################################################################
# main()
###############################################################################

def main():
    ensure_dirs()
    print("[1] Загрузка эталонных образов…")
    templates = load_templates()

    print("[2] Распознаём исходную фразу…")
    recog, boxes, _ = recognise_image(SRC_PATH, templates)
    errs, pct = accuracy(recog, PHRASE_GT)

    print("\nРаспознано :", recog)
    print("Эталон     :", PHRASE_GT)
    print(f"Ошибок      : {errs}/{len(PHRASE_GT)}  |  Точность: {pct:.2f}%")

    # Для наглядности нарисуем прямоугольники
    img_rgb = Image.open(SRC_PATH).convert("RGB")
    draw = ImageDraw.Draw(img_rgb)
    for box in boxes:
        draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="red", width=1)
    img_rgb.save(DST_DIR / "phrase_boxes.bmp")

    print("\n[✓] Завершено. Результаты в", DST_DIR)


if __name__ == "__main__":
    main()
