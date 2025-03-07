from PIL import Image
import numpy as np

image = Image.open('pictures_src/lion.png')
image = image.convert('RGB')

image_array = np.array(image)

R = image_array[:,:,0]
G = image_array[:,:,1]
B = image_array[:,:,2]

Image.fromarray(R).save('pictures_results/R_channel.png')
Image.fromarray(G).save('pictures_results/G_channel.png')
Image.fromarray(B).save('pictures_results/B_channel.png')


def rgb_to_hsi(image_array):
    R = image_array[:, :, 0] / 255.0
    G = image_array[:, :, 1] / 255.0
    B = image_array[:, :, 2] / 255.0

    I = (R + G + B) / 3

    min_rgb = np.minimum(np.minimum(R, G), B)
    S = 1 - 3 * (min_rgb) / (R + G + B + 1e-10)

    numerator = 0.5 * ((R - G) + (R - B))
    denominator = np.sqrt((R - G) ** 2 + (R - B) * (G - B))
    theta = np.arccos(numerator / (denominator + 1e-10))

    H = np.degrees(theta)
    H[B > G] = 360 - H[B > G]

    return H, S, I


H, S, I = rgb_to_hsi(image_array)

Image.fromarray((I * 255).astype(np.uint8)).save('pictures_results/I_component.png')

I_inverted = 255 - (I * 255).astype(np.uint8)

Image.fromarray(I_inverted).save('pictures_results/I_inverted.png')


def stretch_image(image_array, factor):
    new_shape = (int(image_array.shape[0] * factor), int(image_array.shape[1] * factor), image_array.shape[2])
    stretched_image = np.zeros(new_shape, dtype=np.uint8)

    for i in range(new_shape[0]):
        for j in range(new_shape[1]):
            orig_x = int(i / factor)
            orig_y = int(j / factor)
            stretched_image[i, j] = image_array[orig_x, orig_y]

    return stretched_image


stretched_image = stretch_image(image_array, 2)
Image.fromarray(stretched_image).save('pictures_results/stretched_image.png')


def compress_image(image_array, factor):
    new_shape = (int(image_array.shape[0] / factor), int(image_array.shape[1] / factor), image_array.shape[2])
    compressed_image = np.zeros(new_shape, dtype=np.uint8)

    for i in range(new_shape[0]):
        for j in range(new_shape[1]):
            orig_x = int(i * factor)
            orig_y = int(j * factor)
            compressed_image[i, j] = image_array[orig_x, orig_y]

    return compressed_image


compressed_image = compress_image(image_array, 3)
Image.fromarray(compressed_image).save('pictures_results/compressed_image.png')

def resample_image(image_array, M, N):
    stretched = stretch_image(image_array, M)
    resampled = compress_image(stretched, N)
    return resampled

resampled_image = resample_image(image_array, 2, 2)
Image.fromarray(resampled_image).save('pictures_results/resampled_image.png')


def one_step_resample(image_array, factor):
    new_shape = (int(image_array.shape[0] / factor), int(image_array.shape[1] / factor), image_array.shape[2])
    resampled_image = np.zeros(new_shape, dtype=np.uint8)

    for i in range(new_shape[0]):
        for j in range(new_shape[1]):
            orig_x = int(i * factor)
            orig_y = int(j * factor)
            resampled_image[i, j] = image_array[orig_x, orig_y]

    return resampled_image


one_step_resampled_image = one_step_resample(image_array, 2)
Image.fromarray(one_step_resampled_image).save('pictures_results/one_step_resampled_image.png')
