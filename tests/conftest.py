# Copyright (c) Universidad Politecnica Madrid, 2025
# Authors: Alejandro Náger Fernández-Calvo <a.nager@alumnos.upm.es>
# Dates:
#  Creation: Mar. 24, 2025
#  Modification: Apr. 17, 2025
# Documented by: Alejandro Náger Fernández-Calvo <a.nager@alumnos.upm.es>

#tests/conftest.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_DIR = "data/results"
INPUT_IMAGE_DIR = "data/input"
OUTPUT_IMAGE_DIR = "data/output"

results = []

def pytest_configure(config):
    config.results = results  # Permite acceder desde test_lsb.py

def save_difference_map(original_path, stego_path, output_path):
    original = np.array(Image.open(original_path).convert("L")).astype(np.int16)
    stego = np.array(Image.open(stego_path).convert("L")).astype(np.int16)

    diff = np.abs(original - stego).astype(np.uint8)
    amplified_diff = np.clip(diff * 20, 0, 255).astype(np.uint8)  # mayor amplificación

    plt.figure(figsize=(6, 6))
    plt.imshow(amplified_diff, cmap="gray")
    plt.title("Diferencias visuales (escala de grises)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def save_modified_pixel_mask(original_path, stego_path, output_path):
    original = np.array(Image.open(original_path).convert("RGB"))
    stego = np.array(Image.open(stego_path).convert("RGB"))

    # Dónde hay al menos 1 canal modificado
    mask = np.any(original != stego, axis=-1).astype(np.uint8) * 255

    plt.figure(figsize=(6, 6))
    plt.imshow(mask, cmap='Reds')  # Más visible
    plt.title("Zonas modificadas")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def pytest_sessionfinish(session, exitstatus):
    if not results:
        return

    images = [r["image"] + "-" + r["message"] for r in results]
    psnrs = [r["psnr"] for r in results]
    ssims = [r["ssim"] for r in results]
    capacities = [r["capacity"] for r in results]

    def save_plot(y_values, ylabel, title, filename):
        plt.figure(figsize=(10, 5))
        plt.bar(images, y_values, color='cornflowerblue')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel(ylabel)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, filename))
        plt.close()

    save_plot(psnrs, "PSNR (dB)", "PSNR por imagen", "psnr_plot.png")
    save_plot(ssims, "SSIM", "SSIM por imagen", "ssim_plot.png")
    save_plot(capacities, "Capacidad (bits/píxel)", "Capacidad de ocultación", "capacity_plot.png")

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(RESULTS_DIR, "results.csv"), index=False)
    print("\nResultados guardados en data/results/results.csv")

    # Mapas de diferencia y máscaras
        # Mapas de diferencia y máscaras para LSB clásico y LSB optimizado
    for filename in os.listdir(OUTPUT_IMAGE_DIR):
        if any(filename.startswith(p) for p in ("lsb_", "lsbopt_", "lsbadapt_")) and filename.endswith(".png"):
            if filename.startswith("lsb_"):
                prefix = "lsb_"
            elif filename.startswith("lsbopt_"):
                prefix = "lsbopt_"
            elif filename.startswith("lsbadapt_"):
                prefix = "lsbadapt_"
            else:
                continue

            original_filename = filename.replace(prefix, "")
            original_path = os.path.join(INPUT_IMAGE_DIR, original_filename)
            stego_path = os.path.join(OUTPUT_IMAGE_DIR, filename)

            diff_map_path = os.path.join(RESULTS_DIR, filename.replace(".png", "_diffmap.png"))
            mask_path = os.path.join(RESULTS_DIR, filename.replace(".png", "_mask.png"))

            if os.path.exists(original_path):
                save_difference_map(original_path, stego_path, diff_map_path)
                save_modified_pixel_mask(original_path, stego_path, mask_path)

    print("\nMapas de diferencias y máscaras generados en data/results/")

    # Ahora los comparativos por método
    df = pd.read_csv(os.path.join(RESULTS_DIR, "results.csv"))

    # Configuración de estilo
    sns.set(style="whitegrid")

    # Comparativo PSNR
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="image", y="psnr", hue="method", ci=None)
    plt.title("Comparación de PSNR por imagen y método")
    plt.ylabel("PSNR (dB)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "psnr_comparison.png"))
    plt.close()

    # Comparativo SSIM
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="image", y="ssim", hue="method", ci=None)
    plt.title("Comparación de SSIM por imagen y método")
    plt.ylabel("SSIM")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "ssim_comparison.png"))
    plt.close()

    # Comparativo capacidad
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="image", y="capacity", hue="method", ci=None)
    plt.title("Comparación de capacidad (bpp) por imagen y método")
    plt.ylabel("Bits por píxel (bpp)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "capacity_comparison.png"))
    plt.close()

    print("\nComparativos guardados en data/results/:")
    print("  - psnr_comparison.png")
    print("  - ssim_comparison.png")
    print("  - capacity_comparison.png")

    # Comparativo Local MSE
    if "local_mse" in df.columns:
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df, x="image", y="local_mse", hue="method", ci=None, palette="magma")
        plt.title("Comparación de MSE Localizado por imagen y método")
        plt.ylabel("MSE por bloque (media)")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "local_mse_comparison.png"))
        plt.close()

        print("  - local_mse_comparison.png")

    # Comparativo píxeles modificados
    if "pixels_modified_ratio" in df.columns:
        plt.figure(figsize=(10,6))
        sns.barplot(data=df, x="image", y="pixels_modified_ratio", hue="method", ci=None)
        plt.title("Fracción de píxeles modificados por método")
        plt.ylabel("Píxeles modificados (%)")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "pixels_modified_comparison.png"))
        plt.close()
        print("  - pixels_modified_comparison.png")

    if "bits_modified_ratio" in df.columns:
        plt.figure(figsize=(10,6))
        sns.barplot(data=df, x="image", y="bits_modified_ratio", hue="method", ci=None)
        plt.title("Fracción de bits modificados por método")
        plt.ylabel("Bits modificados (%)")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "bits_modified_comparison.png"))
        plt.close()
        print("  - bits_modified_comparison.png")



