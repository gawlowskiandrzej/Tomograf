#pip install pillow
#pip install matplotlib
#pip install opencv-python
#pip install numpy


import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog
from PIL import Image, ImageTk
import io


def loadFile(filename):
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
            sinogram[i, d] = sum(image[y, x] for x, y in points if 0 <= x < w and 0 <= y < h)

    return sinogram

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

class TomographApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Tomograf")
        self.root.geometry("1000x670")

        self.btn_file = tk.Button(root, text="Załaduj obraz", command=self.loadImage)
        self.btn_file.pack(pady=5)

        self.num_detectors = 700
        self.step = 4
        self.angles = np.linspace(0, 360, int(360 / self.step), endpoint=False)

        self.slider = tk.Scale(root, from_=4, to=1, orient=tk.HORIZONTAL, label="Krok interaktywny", command=self.updateImage)
        self.slider.set(4)
        self.slider.pack(pady=10)

        self.btn_generate = tk.Button(root, text="Pokaż sinogram", command=self.showSinogram)
        self.btn_generate.pack(pady=5)

        self.btn_generate = tk.Button(root, text="Pokaż obraz wyjściowy", command=self.showInverseRadon)
        self.btn_generate.pack(pady=5)

        #Ramka do obrazków
        self.frame = tk.Frame(root)
        self.frame.pack()

        titles = ["obraz wejściowy", "sinogram", "obraz wyjściowy"]
        for i, title in enumerate(titles):
            title_lbl = tk.Label(self.frame, text=title, font=("Arial", 12, "bold"))
            title_lbl.grid(row=0, column=i, padx=5, pady=(0, 5))

        self.labels = []
        for i in range(3):
            lbl = tk.Label(self.frame, width=250)
            lbl.grid(row=1, column=i, padx=5, pady=0)
            self.labels.append(lbl)

    def loadImage(self):
        filename = filedialog.askopenfilename()
        self.image = loadFile(filename)
        self.spread = self.image.shape[0]
        self.loadImageToLabel(0, self.image)

    def showSinogram(self):
        for widget in self.labels[1].winfo_children():
            widget.destroy()

        self.sino = []
        self.sinoParts = []
        self.angles_list = []
        self.anglesFin = []

        for i in range(1, 5):
            angles = np.linspace((i - 1) * 90, i * 90, int(90 / self.step), endpoint=False)
            sinogram_part = radon_transform(self.image, angles, self.num_detectors, self.spread)
            self.sino.append(sinogram_part)
            self.angles_list.append(angles)
            self.sinoParts.append(np.concatenate(self.sino, axis=0))
            self.anglesFin.append(np.concatenate(self.angles_list, axis=0))
            print(f"Rekonstrukcja dla sinogramu {angles[0]}° do {angles[-1]}°")
        

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(self.sinoParts[3], cmap='gray', aspect='auto')
        ax.axis('off')

        self.sinogram_canvas = FigureCanvasTkAgg(fig, master=self.labels[1])
        self.sinogram_canvas.draw()
        self.sinogram_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)


    def showInverseRadon(self):
        if not hasattr(self, 'sino') or not hasattr(self, 'angles_list'):
            self.showSinogram()
        self.reconstructedIter = []
        for sino, angles in zip(self.sinoParts, self.anglesFin):
            recon = inverse_radon(sino, angles, self.image.shape[0], self.spread)
            self.reconstructedIter.append(recon)
            print(f"Rekonstrukcja dla kątów {angles[0]} do {angles[-1]}")

        self.loadImageToLabel(2, self.reconstructedIter[3])



    def loadImageToLabel(self, labelNumer, image):
        image_uint8 = (image * 255).astype(np.uint8)
        obraz = Image.fromarray(image_uint8)
        obraz = self.resizeImage(obraz)
        obrazek_tk = ImageTk.PhotoImage(obraz)
        label = self.labels[labelNumer]
        label.configure(image=obrazek_tk)
        label.image = obrazek_tk

    def updateImage(self,num):
        num=int(num)
        
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(self.sinoParts[num-1], cmap='gray', aspect='auto')
        ax.axis('off')  # opcjonalnie

        for widget in self.labels[1].winfo_children():
            widget.destroy()
        # Tworzenie canvas i osadzanie w GUI
        self.sinogram_canvas = FigureCanvasTkAgg(fig, master=self.labels[1])
        self.sinogram_canvas.draw()
        self.sinogram_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
    
        self.loadImageToLabel(2, self.reconstructedIter[num-1])


    def resizeImage(self, obraz, max_width=250, max_height=250):
        original_width, original_height = obraz.size
        width_ratio = max_width / original_width
        height_ratio = max_height / original_height
        scale_ratio = min(width_ratio, height_ratio)
        new_width = int(original_width * scale_ratio)
        new_height = int(original_height * scale_ratio)

        return obraz.resize((new_width, new_height), Image.LANCZOS)

def main():
    root = tk.Tk()
    app = TomographApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
