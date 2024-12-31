# 210229005 - Mert Eren Keküç
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, StringVar, OptionMenu, Button, Label, messagebox
import time 

timing_results = []

def apply_filters():
    global processed_image 
    
    edge_algorithm = edge_var.get()
    corner_algorithm = corner_var.get()

    if loaded_image is None:
        messagebox.showerror("Hata", "Lütfen bir görüntü yükleyin.")
        return

    if edge_algorithm == "Seçiniz" and corner_algorithm == "Seçiniz":
        messagebox.showwarning("Uyarı", "Lütfen kenar veya köşe algoritmalarından en az birini seçin.")
        return

    processed_image = loaded_image.copy()
    gray_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)

    # Gürültü azaltma (Gaussian Blur)
    gray_image = cv2.GaussianBlur(gray_image, (5, 5), 1)

    edge_results = []
    corner_results = []
    start_time = time.time()

    if edge_algorithm != "Seçiniz":
        if edge_algorithm == "Sobel":
            sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
            sobel_combined = cv2.convertScaleAbs(sobel_x + sobel_y)
            edge_results.append(sobel_combined)
        elif edge_algorithm == "Canny":
            edges = cv2.Canny(gray_image, 100, 200)
            edge_results.append(edges)
        elif edge_algorithm == "Laplacian":
            laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
            laplacian_abs = cv2.convertScaleAbs(laplacian)
            edge_results.append(laplacian_abs)

    # Kenarlardan sonra uygula köşeyi
    if corner_algorithm != "Seçiniz":
        if corner_algorithm == "Harris":
            harris = cv2.cornerHarris(np.float32(gray_image), 2, 3, 0.04)
            harris_dilated = cv2.dilate(harris, None) 
            mask = harris_dilated > 0.01 * harris_dilated.max()
            processed_image[mask] = [0, 0, 255] 
            corner_results.append(processed_image.copy())
        elif corner_algorithm == "Shi-Tomasi":
            corners = cv2.goodFeaturesToTrack(gray_image, maxCorners=100, qualityLevel=0.01, minDistance=10)
            for corner in corners:
                x, y = corner.ravel()
                cv2.circle(processed_image, (int(x), int(y)), 5, (0, 255, 0), -1)
            corner_results.append(processed_image.copy())

    end_time = time.time()
    elapsed_time = end_time - start_time

    timing_results.append({
        "edge_algorithm": edge_algorithm,
        "corner_algorithm": corner_algorithm,
        "elapsed_time": elapsed_time
    })

    # Kenar ve köşe birleştir
    if edge_results and corner_results:
        edge_image = edge_results[0]
        corner_image = corner_results[0]

        if edge_image.shape != corner_image.shape:
            corner_image = resize_preserving_aspect(corner_image, edge_image.shape[1], edge_image.shape[0])

        if len(edge_image.shape) == 2:
            edge_image = cv2.cvtColor(edge_image, cv2.COLOR_GRAY2BGR)

        combined_result = cv2.addWeighted(edge_image, 0.5, corner_image, 0.5, 0)
        resized_result = resize_preserving_aspect(combined_result, 800, 600)
        cv2.imshow("Combined Result", resized_result)
        compare_results(edge_image, corner_image, combined_result)

    elif edge_results:
        for i, result in enumerate(edge_results):
            resized_result = resize_preserving_aspect(result, 800, 600)
            cv2.imshow(f"Edge Result {i+1}", resized_result)

    elif corner_results:
        for i, result in enumerate(corner_results):
            resized_result = resize_preserving_aspect(result, 800, 600)
            cv2.imshow(f"Corner Result {i+1}", resized_result)

    print(f"Kenar: {edge_algorithm}, Köşe: {corner_algorithm}, Süre: {elapsed_time:.4f} saniye")

def load_image():
    global loaded_image
    file_path = filedialog.askopenfilename(
        title="Bir görüntü dosyası seçin",
        filetypes=[("Image Files", "*.jpg;*.png;*.jpeg;*.bmp;*.tiff"), ("All Files", "*.*")]
    )
    if not file_path:
        print("Hiçbir dosya seçilmedi!")
        return

    loaded_image = cv2.imread(file_path)

    if loaded_image is None:
        messagebox.showerror("Hata", f"Görüntü yüklenemedi! Dosya yolu: {file_path}")
        return

    zoomed_image = resize_image(loaded_image, 0.5)
    cv2.imshow("Yuklenen Goruntu", zoomed_image)

def resize_image(image, scale_percent):
    width = int(image.shape[1] * scale_percent)
    height = int(image.shape[0] * scale_percent)
    dim = (width, height)
    
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized

def resize_preserving_aspect(image, max_width, max_height):
    original_height, original_width = image.shape[:2]
    aspect_ratio = original_width / original_height

    if max_width / aspect_ratio <= max_height:
        new_width = max_width
        new_height = int(max_width / aspect_ratio)
    else:
        new_height = max_height
        new_width = int(max_height * aspect_ratio)

    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

def save_image():
    if processed_image is None:
        messagebox.showerror("Hata", "Henüz işlenmiş bir görüntü yok.")
        return

    file_path = filedialog.asksaveasfilename(defaultextension=".png")
    if file_path:
        cv2.imwrite(file_path, processed_image)
        messagebox.showinfo("Bilgi", f"Görüntü başarıyla kaydedildi: {file_path}")

def compare_results(edge_image, corner_image, combined_image):
    combined_height = max(edge_image.shape[0], corner_image.shape[0], combined_image.shape[0])
    combined_width = edge_image.shape[1] + corner_image.shape[1] + combined_image.shape[1]
    comparison_image = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)

    comparison_image[:edge_image.shape[0], :edge_image.shape[1]] = edge_image
    comparison_image[:corner_image.shape[0], edge_image.shape[1]:edge_image.shape[1] + corner_image.shape[1]] = corner_image
    comparison_image[:combined_image.shape[0], edge_image.shape[1] + corner_image.shape[1]:] = combined_image

    resized_comparison = resize_preserving_aspect(comparison_image, 1200, 400)
    cv2.imshow("Comparison Result", resized_comparison)

root = tk.Tk()
root.title("Kenar ve Köşe Tespit Sistemi")

load_button = Button(root, text="Görüntü Yükle", command=load_image)
load_button.grid(row=0, column=0, padx=10, pady=10)


edge_var = StringVar(root)
edge_var.set("Seçiniz")
edge_options = ["Seçiniz", "Sobel", "Canny", "Laplacian"]
edge_menu = OptionMenu(root, edge_var, *edge_options)
edge_menu.grid(row=1, column=0, padx=10, pady=10)
edge_label = Label(root, text="Kenar Algoritması")
edge_label.grid(row=1, column=1)

corner_var = StringVar(root)
corner_var.set("Seçiniz")
corner_options = ["Seçiniz", "Harris", "Shi-Tomasi"]
corner_menu = OptionMenu(root, corner_var, *corner_options)
corner_menu.grid(row=2, column=0, padx=10, pady=10)
corner_label = Label(root, text="Köşe Algoritması")
corner_label.grid(row=2, column=1)

apply_button = Button(root, text="Uygula", command=apply_filters)
apply_button.grid(row=3, column=0, padx=10, pady=10)

save_button = Button(root, text="Kaydet", command=save_image)
save_button.grid(row=3, column=1, padx=10, pady=10)

loaded_image = None
processed_image = None
root.mainloop()
