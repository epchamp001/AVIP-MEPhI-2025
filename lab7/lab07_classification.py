import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

"""
Лабораторная 7 — распознавание фразы по шаблонам (вариант 21, русские заглавные).

Изменения:
• добавлена генерация файла `best_hypotheses.txt`, где для каждого символа
  (без учёта пробелов) выводится лучшая гипотеза и значение IoU.
• код очищен: используется `normalize_bin()` для центрирования + масштабирования.
• метрика сходства — коэффициент Жаккара (IoU).

Запуск одна команда:
    python lab7_feature_classification.py
"""


SRC_PATH     = Path("pictures_src/phrase.bmp")
ALPHABET_DIR = Path("alphabet")
DST_DIR      = Path("pictures_results")
PHRASE_GT    = "БЕРЕГИТЕ В СЕБЕ ЧЕЛОВЕКА"
SIZE         = (64, 64)

ALPHABET = list("АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ")

os.makedirs(DST_DIR, exist_ok=True)


def to_binary(img_or_path) -> np.ndarray:
    """Возвращает бинарное изображение 0/1 (1 — чёрный)."""
    img = Image.open(img_or_path) if isinstance(img_or_path, (str, Path)) else img_or_path
    img = img.convert("L")  # градации серого
    arr = np.array(img)
    return (arr < 128).astype(np.uint8)


def normalize_bin(arr: np.ndarray, size: tuple[int, int] = SIZE) -> np.ndarray:
    """Плотно обрезает символ, центрирует на квадратном холсте, масштабирует."""
    ys, xs = np.nonzero(arr)
    if ys.size == 0:
        return np.zeros(size, dtype=np.uint8)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    crop = arr[y0:y1 + 1, x0:x1 + 1]

    h, w = crop.shape
    side = max(h, w)
    canvas = np.zeros((side, side), dtype=np.uint8)
    y_off = (side - h) // 2
    x_off = (side - w) // 2
    canvas[y_off:y_off + h, x_off:x_off + w] = crop

    pil = Image.fromarray(canvas * 255)  # обратно 0/255
    pil = pil.resize(size, Image.NEAREST)
    res = np.array(pil)
    return (res < 128).astype(np.uint8)


def segment_by_profiles(bin_img: np.ndarray, empty_thresh: int = 1):
    """Возвращает bounding-box'ы символов, без пробелов."""
    h, w = bin_img.shape
    vert = bin_img.sum(axis=0)
    splits, in_char = [], False

    for x, v in enumerate(vert):
        if not in_char and v > empty_thresh:
            in_char, x0 = True, x
        elif in_char and v <= empty_thresh:
            splits.append((x0, x - 1))
            in_char = False
    if in_char:
        splits.append((x0, w - 1))

    boxes = []
    for x0, x1 in splits:
        slice_ = bin_img[:, x0:x1 + 1]
        horiz = slice_.sum(axis=1)
        ys = np.where(horiz > empty_thresh)[0]
        if ys.size:
            boxes.append((x0, ys[0], x1, ys[-1]))
    return boxes


def split_wide_boxes(boxes, bin_img, factor: float = 1.4):
    """Если бокс слишком широкий (клейкие буквы), делит по минимуму профиля."""
    widths = [x1 - x0 + 1 for x0, _, x1, _ in boxes]
    if not widths:
        return boxes
    avg_w = sum(widths) / len(widths)

    out = []
    for (x0, y0, x1, y1), w in zip(boxes, widths):
        if w > avg_w * factor:
            sub = bin_img[y0:y1 + 1, x0:x1 + 1]
            vert = sub.sum(axis=0)
            m = max(w // 8, 1)
            local = vert[m:-m]
            if local.size:
                cut_off = np.argmin(local) + m
                cut = x0 + cut_off
                out += [(x0, y0, cut, y1), (cut + 1, y0, x1, y1)]
                continue
        out.append((x0, y0, x1, y1))
    return out


def gap_is_space(prev_box, curr_box, ratio=0.5):
    if prev_box is None:
        return False
    gap = curr_box[0] - prev_box[2]
    return gap > (prev_box[2] - prev_box[0]) * ratio


def load_templates():
    tpls = []
    for ch in ALPHABET:
        path = ALPHABET_DIR / f"{ch}.bmp"
        bin_img = to_binary(path)
        tpl = normalize_bin(bin_img).astype(bool)
        tpls.append(tpl)
    return np.stack(tpls, axis=0), ALPHABET  # shape: (33, 64, 64)


def compute_iou_batch(tpl_stack: np.ndarray, sub: np.ndarray) -> np.ndarray:
    sub_b = sub.astype(bool)
    inter = np.logical_and(tpl_stack, sub_b).sum(axis=(1, 2))
    union = np.logical_or(tpl_stack, sub_b).sum(axis=(1, 2))
    return inter / np.where(union == 0, 1, union)


def recognise_image(path: Path, tpl_stack: np.ndarray, keys: list[str]):
    bin_img = to_binary(path)
    boxes = segment_by_profiles(bin_img)
    boxes = split_wide_boxes(boxes, bin_img)
    boxes.sort(key=lambda b: b[0])

    result = ""
    letters, scores = [], []
    prev = None

    for x0, y0, x1, y1 in boxes:
        sub = bin_img[y0:y1 + 1, x0:x1 + 1]
        sub = normalize_bin(sub).astype(bool)

        ious = compute_iou_batch(tpl_stack, sub)
        best_idx = int(ious.argmax())
        best_ch = keys[best_idx]
        best_score = float(ious[best_idx])

        if gap_is_space(prev, (x0, y0, x1, y1)):
            result += " "
        result += best_ch
        letters.append(best_ch)
        scores.append(best_score)
        prev = (x0, y0, x1, y1)

    return result, boxes, letters, scores

def accuracy(pred: str, gt: str):
    m = max(len(pred), len(gt))
    errs = sum(1 for a, b in zip(pred.ljust(m), gt.ljust(m)) if a != b)
    return errs, 100 * (1 - errs / m)


def main():
    print("[1] Загрузка шаблонов…")
    tpl_stack, keys = load_templates()

    print("[2] Сегментация и распознавание…")
    recog, boxes, letters, scores = recognise_image(SRC_PATH, tpl_stack, keys)
    errs, pct = accuracy(recog, PHRASE_GT)

    print(f"\nРаспознано : {recog}")
    print(f"Эталон     : {PHRASE_GT}")
    print(f"Ошибок     : {errs}/{len(PHRASE_GT)}  |  Точность: {pct:.2f}%")

    img = Image.open(SRC_PATH).convert("RGB")
    draw = ImageDraw.Draw(img)
    for box in boxes:
        draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="red", width=1)
    img.save(DST_DIR / "phrase_boxes_fixed.bmp")

    hyp_path = DST_DIR / "best_hypotheses.txt"
    with hyp_path.open("w", encoding="utf-8") as f:
        f.write("Выводятся лучшие гипотезы\n\n")
        idx = 1
        for ch, sc in zip(letters, scores):
            # пропускаем пробелы
            if ch == " ":
                continue
            f.write(f"{idx:2d}: '{ch}' - {sc:.6f}\n")
            idx += 1
    print("[✓] Файл с гипотезами →", hyp_path)
    print("[✓] Готово, остальные результаты — в", DST_DIR)


if __name__ == "__main__":
    main()
