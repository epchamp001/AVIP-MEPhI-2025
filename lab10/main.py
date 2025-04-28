# -*- coding: utf-8 -*-
"""
main.py — Лабораторная работа №10: Обработка голоса
"""
import os
import glob
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# ─── ПУТИ ───────────────────────────────────────────────────────────────────────
SRC_DIR     = 'src'
RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─── ФУНКЦИИ ────────────────────────────────────────────────────────────────────

def plot_spectrogram(y, sr, title, outpath):
    D = librosa.stft(y, n_fft=2048, hop_length=512, window='hann')
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    plt.figure(figsize=(8,4))
    librosa.display.specshow(S_db, sr=sr, hop_length=512,
                             x_axis='time', y_axis='log', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def min_max_frequency(y, sr, threshold_db=-60):
    # усреднённый спектр
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    mean_spec = S_db.mean(axis=1)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    mask = mean_spec > threshold_db
    if not mask.any():
        return 0.0, 0.0
    fmin = freqs[mask].min()
    fmax = freqs[mask].max()
    return fmin, fmax


def estimate_f0_and_overtones(y, sr, fmin=50, fmax=800):
    # контур основного тона
    f0 = librosa.yin(y, fmin=fmin, fmax=fmax, sr=sr, frame_length=2048, hop_length=512)
    f0_med = np.nanmedian(f0)
    # спектр амплитудный
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    spec_avg = S.mean(axis=1)
    # поиск обертонов
    harmonics = []
    k = 1
    while True:
        h_freq = f0_med * k
        if h_freq >= sr/2:
            break
        # найти ближайший индекс частоты
        idx = np.argmin(np.abs(freqs - h_freq))
        # если энергия выше 50% от пика
        if spec_avg[idx] > 0.5 * spec_avg.max():
            harmonics.append(h_freq)
        k += 1
    return f0_med, harmonics


def estimate_formants(y, sr, n_formants=3, lpc_order=16):
    # LPC-коэффициенты
    a = librosa.lpc(y, order=lpc_order)
    roots = np.roots(a)
    angles = np.angle(roots)
    freqs = angles * sr / (2 * np.pi)
    # берем только положительные частоты и real>0
    freqs = freqs[freqs>0]
    freqs = np.sort(freqs)
    return freqs[:n_formants]

# ─── main() ───────────────────────────────────────────────────────────────────

def main():
    files = glob.glob(os.path.join(SRC_DIR, '*.wav'))
    report = []

    for path in files:
        name = os.path.splitext(os.path.basename(path))[0]
        y, sr = librosa.load(path, sr=None, mono=True)
        print(f"Processing {name} (sr={sr}, len={len(y)/sr:.2f}s)")

        # 2) спектрограмма
        spec_path = os.path.join(RESULTS_DIR, f'spec_{name}.png')
        plot_spectrogram(y, sr, f'Spectrogram: {name}', spec_path)

        # 3) min/max freq
        fmin, fmax = min_max_frequency(y, sr)

        # 4) основной тон и обертоны
        f0_med, harmonics = estimate_f0_and_overtones(y, sr)

        # 5) форманты
        formants = estimate_formants(y, sr)

        # сохраняем метрики
        report.append({
            'name': name,
            'fmin': fmin,
            'fmax': fmax,
            'f0': f0_med,
            'num_overtones': len(harmonics),
            'formants': formants.tolist()
        })

    # Запись отчёта в файл
    with open(os.path.join(RESULTS_DIR, 'report.txt'), 'w', encoding='utf-8') as f:
        for item in report:
            f.write(f"File: {item['name']}\n")
            f.write(f"Min freq: {item['fmin']:.1f} Hz, Max freq: {item['fmax']:.1f} Hz\n")
            f.write(f"Fundamental (median): {item['f0']:.1f} Hz, Overtones count: {item['num_overtones']}\n")
            f.write(f"Formants: {', '.join(f'{freq:.1f}' for freq in item['formants'])} Hz\n")
            f.write("\n")
    print(f"Report saved to {os.path.join(RESULTS_DIR, 'report.txt')}")

if __name__ == '__main__':
    main()
