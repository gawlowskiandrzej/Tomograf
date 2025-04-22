import numpy as np
import cv2
import matplotlib.pyplot as plt

def loadImage(filename):
    file = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    if file is None:
        raise FileNotFoundError(f"Nie znaleziono pliku: {filename}")
    return file.astype(np.float32) / 255.0

def bresenham_line(x0, y0, x1, y1):
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

    return points

def radon_transform(image, angles, num_detectors, spread):
    h, w = image.shape
    center_x, center_y = w // 2, h // 2
    sinogram = np.zeros((len(angles), num_detectors))

    for i, alpha in enumerate(angles):
        alpha_rad = np.deg2rad(alpha)
        for d in range(num_detectors):
            offset = (d - num_detectors // 2) * spread / num_detectors
            x_e = int(center_x + np.cos(alpha_rad) * (w // 2) + np.sin(alpha_rad) * offset)
            y_e = int(center_y + np.sin(alpha_rad) * (h // 2) - np.cos(alpha_rad) * offset)
            x_d = int(center_x - np.cos(alpha_rad) * (w // 2) + np.sin(alpha_rad) * offset)
            y_d = int(center_y - np.sin(alpha_rad) * (h // 2) - np.cos(alpha_rad) * offset)

            points = bresenham_line(x_e, y_e, x_d, y_d)
            sinogram[i, d] = sum([image[y, x] for x, y in points if 0 <= x < w and 0 <= y < h])

    return sinogram

def create_filter_kernel(size=21):
    kernel = np.zeros(size)
    center = size // 2
    for k in range(-center, center + 1):
        if k == 0:
            kernel[k + center] = 1
        elif k % 2 == 0:
            kernel[k + center] = 0
        else:
            kernel[k + center] = -4 / (np.pi ** 2 * k ** 2)
    return kernel

def apply_filter_to_sinogram(sinogram, kernel):
    filtered_sino = np.zeros_like(sinogram)
    for i in range(sinogram.shape[0]):
        filtered_sino[i, :] = np.convolve(sinogram[i, :], kernel, mode='same')
    return filtered_sino

def inverse_radon(sinogram, angles, img_size, spread):
    reconstructed = np.zeros((img_size, img_size))
    center = img_size // 2

    for i, alpha in enumerate(angles):
        alpha_rad = np.deg2rad(alpha)
        for d in range(sinogram.shape[1]):
            value = sinogram[i, d]
            offset = (d - sinogram.shape[1] // 2) * spread / sinogram.shape[1]
            x_e = int(center + np.cos(alpha_rad) * center + np.sin(alpha_rad) * offset)
            y_e = int(center + np.sin(alpha_rad) * center - np.cos(alpha_rad) * offset)
            x_d = int(center - np.cos(alpha_rad) * center + np.sin(alpha_rad) * offset)
            y_d = int(center - np.sin(alpha_rad) * center - np.cos(alpha_rad) * offset)

            points = bresenham_line(x_e, y_e, x_d, y_d)
            for x, y in points:
                if 0 <= x < img_size and 0 <= y < img_size:
                    reconstructed[y, x] += value / len(angles)

    max_val = np.max(reconstructed)
    if max_val > 0:
        reconstructed /= max_val

    return reconstructed

def compute_rmse(original, reconstructed):
    mask = (original > 0) | (reconstructed > 0)
    diff = original[mask] - reconstructed[mask]
    return np.sqrt(np.mean(diff ** 2))

def main():
    image = loadImage("Shepp_logan.jpg")
    step = 1
    angles = np.linspace(0, 360, int(360 / step), endpoint=False)
    num_detectors = 300
    spread = image.shape[0]

    sinogram = radon_transform(image, angles, num_detectors, spread)

    # Stworzenie i zastosowanie filtra
    kernel = create_filter_kernel(21)
    filtered_sinogram = apply_filter_to_sinogram(sinogram, kernel)

    # Odtworzenie obrazu z przefiltrowanego sinogramu
    reconstructed_image = inverse_radon(sinogram, angles, image.shape[0], spread)
    reconstructed_image2 = inverse_radon(filtered_sinogram, angles, image.shape[0], spread)

    # Obliczenie RMSE
    rmse = compute_rmse(image, reconstructed_image)
    print(f"RMSE: {rmse:.4f}")

    # Wizualizacja
    fig, ax = plt.subplots(1, 5, figsize=(15, 5))
    ax[0].imshow(image, cmap="gray")
    ax[0].set_title("Obraz wejściowy")
    ax[0].axis("off")

    ax[1].imshow(sinogram, cmap="gray", aspect='auto')
    ax[1].set_title("Filtrowany Sinogram")
    ax[1].axis("off")


    ax[2].imshow(filtered_sinogram, cmap="gray", aspect='auto')
    ax[2].set_title("Filtrowany Sinogram")
    ax[2].axis("off")

    ax[3].imshow(reconstructed_image, cmap="gray")
    ax[3].set_title("Odtworzony Obraz")
    ax[3].axis("off")

    ax[4].imshow(reconstructed_image2, cmap="gray")
    ax[4].set_title("Odtworzony Obraz")
    ax[4].axis("off")

    plt.show()

if __name__ == "__main__":
    main()
