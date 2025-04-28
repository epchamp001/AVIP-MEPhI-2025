import os
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

# ──────────────────────────────────────────────────────────────────────────────
# Константы
# ──────────────────────────────────────────────────────────────────────────────

SRC_PATH = "pictures_src/phrase.bmp"
DST_DIR   = "pictures_results"
os.makedirs(DST_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# Утилиты
# ──────────────────────────────────────────────────────────────────────────────

def to_binary(path: str) -> np.ndarray:
    """
    Приводит картинку к numpy-массиву 0/1:
    1 — чёрный пиксель (текст), 0 — фон.
    """
    img = Image.open(path).convert("L")       # градации серого
    arr = np.array(img)
    return (arr < 128).astype(np.uint8)       # порог 128

def profiles(bin_img: np.ndarray):
    """Возвращает (горизонтальный, вертикальный) профили."""
    return bin_img.sum(axis=1), bin_img.sum(axis=0)

# ──────────────────────────────────────────────────────────────────────────────
# Сегментация
# ──────────────────────────────────────────────────────────────────────────────

def segment_by_profiles(bin_img: np.ndarray, empty_thresh: int = 2):
    """
    Разбивает строку на буквы по вертикальному профилю.
    empty_thresh — сколько чёрных пикселей в столбце считаем «пустым».
    """
    h, w = bin_img.shape
    vert = bin_img.sum(axis=0)
    splits, in_char = [], False

    for x, v in enumerate(vert):
        if not in_char and v > empty_thresh:
            in_char, x0 = True, x
        elif in_char and v <= empty_thresh:
            splits.append((x0, x-1))
            in_char = False
    if in_char:
        splits.append((x0, w-1))

    boxes = []
    for x0, x1 in splits:
        slice_ = bin_img[:, x0:x1+1]
        horiz = slice_.sum(axis=1)
        ys = np.where(horiz > empty_thresh)[0]
        if ys.size:
            boxes.append((x0, ys[0], x1, ys[-1]))
    return boxes

def split_wide_boxes(boxes: list[tuple], bin_img: np.ndarray, factor: float = 1.4):
    """
    Если бокс шире, чем factor * средняя ширина всех боксов,
    разрезаем его по локальному минимуму вертикального профиля.
    """
    widths = [x1 - x0 + 1 for x0,_,x1,_ in boxes]
    if not widths:
        return boxes
    avg_w = sum(widths) / len(widths)

    out = []
    for (x0,y0,x1,y1), w in zip(boxes, widths):
        if w > avg_w * factor:
            sub = bin_img[y0:y1+1, x0:x1+1]
            vert = sub.sum(axis=0)
            margin = max(w // 10, 1)
            local = vert[margin:-margin]
            if local.size:
                cut_rel = int(np.argmin(local)) + margin
                cut = x0 + cut_rel
                out += [(x0,y0,cut,y1), (cut+1,y0,x1,y1)]
                continue
        out.append((x0,y0,x1,y1))
    return out

# ──────────────────────────────────────────────────────────────────────────────
# Сохранение профилей букв (с простыми индексами)
# ──────────────────────────────────────────────────────────────────────────────

def save_letter_profiles(bin_img: np.ndarray, boxes: list[tuple]):
    for idx, (x0,y0,x1,y1) in enumerate(boxes):
        # вырезаем кусочек
        patch = bin_img[y0:y1+1, x0:x1+1]
        # сохраняем BMP
        char_img = Image.fromarray((1 - patch) * 255)
        bmp_name = f"{idx:02d}.bmp"
        char_img.save(os.path.join(DST_DIR, bmp_name))
        # сохраняем профили
        h_prof, v_prof = profiles(patch)
        txt_name = f"{idx:02d}.txt"
        with open(os.path.join(DST_DIR, txt_name), "w", encoding="utf-8") as f:
            f.write("horizontal:\n" + " ".join(map(str, h_prof.tolist())) + "\n")
            f.write("vertical:\n"   + " ".join(map(str, v_prof.tolist())))

# ──────────────────────────────────────────────────────────────────────────────
# Визуализация боксов
# ──────────────────────────────────────────────────────────────────────────────

def draw_boxes(path: str, boxes: list[tuple]):
    img = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for x0,y0,x1,y1 in boxes:
        draw.rectangle([ (x0,y0), (x1,y1) ], outline="red", width=2)
    img.save(os.path.join(DST_DIR, "phrase_boxes_fixed.bmp"))

# ──────────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    bin_img = to_binary(SRC_PATH)

    # сохраняем общие профили
    h, v = profiles(bin_img)
    np.savetxt(os.path.join(DST_DIR, "horiz_profile.txt"), h, fmt="%d")
    np.savetxt(os.path.join(DST_DIR, "vert_profile.txt"), v, fmt="%d")

    # сегментация + разбивка «КА»
    boxes = segment_by_profiles(bin_img, empty_thresh=2)
    boxes = split_wide_boxes(boxes, bin_img, factor=1.4)

    # рисуем рамки и сохраняем буквы с профилями
    draw_boxes(SRC_PATH, boxes)
    save_letter_profiles(bin_img, boxes)

    print(f"Сегментировано символов: {len(boxes)}")
    print(f"Результаты в {DST_DIR}")

if __name__ == "__main__":
    main()
