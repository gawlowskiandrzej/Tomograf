#pip install pillow matplotlib opencv-python numpy pydicom scikit-image

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from pydicom.uid import CTImageStorage
import tkinter as tk
from tkinter import messagebox, filedialog
from PIL import Image, ImageTk
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import generate_uid, ExplicitVRLittleEndian
from skimage.util import img_as_ubyte
from skimage.exposure import rescale_intensity
import datetime


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

def compute_rmse(original, reconstructed):
    mask = (original > 0) | (reconstructed > 0)
    diff = original[mask] - reconstructed[mask]
    return np.sqrt(np.mean(diff ** 2))

class TomographApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Tomograf")
        self.root.geometry("1000x813")

        self.num_detectors = tk.IntVar(value=700)
        self.step = tk.IntVar(value=4)
        
        self.invradonFilter = None
        self.sinoFilter = None
        self.Filter = False
        self.rmse = tk.StringVar(value="0.0")

        # Nowa ramka na przyciski w jednej linii
        self.buttons_frame = tk.Frame(root)
        self.buttons_frame.pack(pady=5)

        self.btn_file = tk.Button(self.buttons_frame, text="Załaduj obraz", command=self.loadImage)
        self.btn_file.grid(row=0, column=0, padx=5)

        self.btn_generate = tk.Button(self.buttons_frame, text="Pokaż sinogram", command=self.showSinogram)
        self.btn_generate.grid(row=0, column=1, padx=5)

        self.btn_recon = tk.Button(self.buttons_frame, text="Pokaż obraz wyjściowy", command=self.showInverseRadon)
        self.btn_recon.grid(row=0, column=2, padx=5)

        self.btn_recon = tk.Button(self.buttons_frame, text="Pokaż z filtrem", command=self.showFilter)
        self.btn_recon.grid(row=0, column=3, padx=5)

        tk.Label(self.buttons_frame, text="RMSE").grid(row=0, column=4, sticky="w")
        tk.Label(self.buttons_frame, textvariable=self.rmse).grid(row=0, column=5, sticky="w")

        self.input_frame = tk.Frame(root)
        self.input_frame.pack(pady=5)

        tk.Label(self.input_frame, text="Liczba detektorów").grid(row=0, column=0, sticky="w")
        tk.Entry(self.input_frame, textvariable=self.num_detectors).grid(row=0, column=1)

        tk.Label(self.input_frame, text="delta alfa").grid(row=1, column=0, sticky="w")
        tk.Entry(self.input_frame, textvariable=self.step).grid(row=1, column=1)

        self.slider = tk.Scale(root, from_=4, to=1, orient=tk.HORIZONTAL, label="Krok interaktywny", command=self.updateImage)
        self.slider.set(4)
        self.slider.pack(pady=10)

        self.btn_save_dicom = tk.Button(root, text="Zapisz jako DICOM", command=self.saveAsDicom)
        self.btn_save_dicom.pack(pady=5)

        self.patient_data_frame = tk.LabelFrame(root, text="Dane pacjenta", padx=10, pady=10)
        self.patient_data_frame.pack(pady=5)

        self.patient_name_var = tk.StringVar()
        self.patient_id_var = tk.StringVar()
        self.study_date_var = tk.StringVar()
        self.image_comments_var = tk.StringVar()

        fields = [
            ("Imię i nazwisko", self.patient_name_var),
            ("ID pacjenta", self.patient_id_var),
            ("Data badania (YYYYMMDD)", self.study_date_var),
            ("Komentarz", self.image_comments_var),
        ]

        for i, (label_text, var) in enumerate(fields):
            tk.Label(self.patient_data_frame, text=label_text).grid(row=i, column=0, sticky="w")
            tk.Entry(self.patient_data_frame, textvariable=var).grid(row=i, column=1)

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
            angles = np.linspace((i - 1) * 90, i * 90, int(90 / self.step.get()), endpoint=False)
            sinogram_part = radon_transform(self.image, angles, self.num_detectors.get(), self.spread)
            self.sino.append(sinogram_part)
            self.angles_list.append(angles)
            self.sinoParts.append(np.concatenate(self.sino, axis=0))
            self.anglesFin.append(np.concatenate(self.angles_list, axis=0))
            print(f"Rekonstrukcja dla sinogramu {angles[0]}° do {angles[-1]}°")

        self.sinoFilter = apply_filter_to_sinogram(self.sinoParts[3], create_filter_kernel(21))
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
        print(f"Rekonsturkcja z filtrem")
        self.invradonFilter = inverse_radon(self.sinoFilter, self.anglesFin[-1], self.image.shape[0], self.spread)
        self.rmse.set(compute_rmse(self.image, self.reconstructedIter[3]))
        self.loadImageToLabel(2, self.reconstructedIter[3])
        self.Filter = False

    def updateImage(self, num):
        if hasattr(self, 'sino') or hasattr(self, 'sinoParts'):
            num = int(num)
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(self.sinoParts[num - 1], cmap='gray', aspect='auto')
            ax.axis('off')
            for widget in self.labels[1].winfo_children():
                widget.destroy()
            self.sinogram_canvas = FigureCanvasTkAgg(fig, master=self.labels[1])
            self.sinogram_canvas.draw()
            self.sinogram_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
            self.loadImageToLabel(2, self.reconstructedIter[num - 1])
        
    def loadImageToLabel(self, labelNumer, image):
        image_uint8 = (image * 255).astype(np.uint8)
        obraz = Image.fromarray(image_uint8)
        obraz = self.resizeImage(obraz)
        obrazek_tk = ImageTk.PhotoImage(obraz)
        label = self.labels[labelNumer]
        label.configure(image=obrazek_tk)
        label.image = obrazek_tk

    def resizeImage(self, obraz, max_width=250, max_height=250):
        original_width, original_height = obraz.size
        width_ratio = max_width / original_width
        height_ratio = max_height / original_height
        scale_ratio = min(width_ratio, height_ratio)
        new_width = int(original_width * scale_ratio)
        new_height = int(original_height * scale_ratio)
        return obraz.resize((new_width, new_height), Image.LANCZOS)
    
    def showFilter(self):
        if hasattr(self, 'sino') or hasattr(self, 'sinoParts'):
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(self.sinoFilter, cmap='gray', aspect='auto')
            ax.axis('off')
            for widget in self.labels[1].winfo_children():
                widget.destroy()
            self.sinogram_canvas = FigureCanvasTkAgg(fig, master=self.labels[1])
            self.sinogram_canvas.draw()
            self.sinogram_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
            self.rmse.set(compute_rmse(self.image, self.invradonFilter))
            self.loadImageToLabel(2, self.invradonFilter)
            self.Filter = True
        else:
            messagebox.showerror("Błąd", "Wygeneruj najpierw sinogram")

    def saveAsDicom(self):
        if not hasattr(self, 'reconstructedIter'):
            messagebox.showerror("Błąd", "Brak obrazu do zapisania. Najpierw wygeneruj obraz wyjściowy.")
            return

        file_path = filedialog.asksaveasfilename(defaultextension=".dcm", filetypes=[("DICOM files", "*.dcm")])
        if not file_path:
            return

        patient_data = {
            "PatientName": self.patient_name_var.get() or "Anonimowy",
            "PatientID": self.patient_id_var.get() or "000000",
            "ImageComments": self.image_comments_var.get() or "",
            "StudyDate": self.study_date_var.get() or datetime.datetime.now().strftime("%Y%m%d")
        }
        image_to_save = None
        if self.Filter:
            image_to_save = self.invradonFilter
        else:
            image_to_save = self.reconstructedIter[int(self.slider.get()) - 1]
        self._save_dicom_file(file_path, image_to_save, patient_data)
        messagebox.showinfo("Sukces", f"Obraz zapisany jako DICOM: {file_path}")

    def _save_dicom_file(self, file_name, img, patient_data):
        img_converted = img_as_ubyte(rescale_intensity(img, out_range=(0.0, 1.0)))

        meta = Dataset()
        meta.MediaStorageSOPClassUID = CTImageStorage

        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian

        ds = FileDataset(None, {}, preamble=b"\0" * 128)
        ds.file_meta = meta
        ds.is_little_endian = True
        ds.is_implicit_VR = False

        ds.SOPClassUID = meta.MediaStorageSOPClassUID
        ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        ds.PatientName = patient_data["PatientName"]
        ds.PatientID = patient_data["PatientID"]
        ds.ImageComments = patient_data["ImageComments"]
        ds.StudyDate = patient_data["StudyDate"]
        ds.StudyTime = datetime.datetime.now().strftime("%H%M%S")

        ds.Modality = "CT"
        ds.SeriesInstanceUID = generate_uid()
        ds.StudyInstanceUID = generate_uid()
        ds.FrameOfReferenceUID = generate_uid()

        ds.BitsStored = 8
        ds.BitsAllocated = 8
        ds.SamplesPerPixel = 1
        ds.HighBit = 7

        ds.Rows, ds.Columns = img_converted.shape
        ds.ImagesInAcquisition = 1
        ds.InstanceNumber = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.PixelRepresentation = 0
        ds.ImageType = r"ORIGINAL\PRIMARY\AXIAL"

        ds.PixelData = img_converted.tobytes()
        ds.save_as(file_name, write_like_original=False)


def main():
    root = tk.Tk()
    app = TomographApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
