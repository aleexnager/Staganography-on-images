# Copyright (c) Universidad Politecnica Madrid, 2025
# Authors: Alejandro Náger Fernández-Calvo <a.nager@alumnos.upm.es>
# Dates:
#  Creation: Mar. 24, 2025
#  Modification: Mar. 24, 2025
# Documented by: Alejandro Náger Fernández-Calvo <a.nager@alumnos.upm.es>

# generate_differences.py

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

INPUT_IMAGE_DIR = "data/input"
OUTPUT_IMAGE_DIR = "data/output"
RESULTS_DIR = "data/results"

os.makedirs(RESULTS_DIR, exist_ok=True)

def save_difference_map(original_path, stego_path, output_path):
    original = np.array(Image.open(original_path).convert("RGB")).astype(np.int16)
    stego = np.array(Image.open(stego_path).convert("RGB")).astype(np.int16)

    diff = np.abs(original - stego).astype(np.uint8)
    amplified_diff = np.clip(diff * 10, 0, 255).astype(np.uint8)

    plt.figure(figsize=(6, 6))
    plt.imshow(amplified_diff)
    plt.title("Mapa de diferencias (amplificadas)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def save_modified_pixel_mask(original_path, stego_path, output_path):
    original = np.array(Image.open(original_path).convert("RGB"))
    stego = np.array(Image.open(stego_path).convert("RGB"))

    mask = np.any(original != stego, axis=-1).astype(np.uint8) * 255

    plt.figure(figsize=(6, 6))
    plt.imshow(mask, cmap='hot')
    plt.title("Zonas modificadas")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def generate_all_differences():
    for filename in os.listdir(OUTPUT_IMAGE_DIR):
        if filename.startswith("lsb_") and filename.endswith(".png"):
            original_filename = filename.replace("lsb_", "")
            original_path = os.path.join(INPUT_IMAGE_DIR, original_filename)
            stego_path = os.path.join(OUTPUT_IMAGE_DIR, filename)

            diff_map_path = os.path.join(RESULTS_DIR, filename.replace(".png", "_diffmap.png"))
            mask_path = os.path.join(RESULTS_DIR, filename.replace(".png", "_mask.png"))

            if os.path.exists(original_path):
                print(f"Procesando diferencias para {filename}...")
                save_difference_map(original_path, stego_path, diff_map_path)
                save_modified_pixel_mask(original_path, stego_path, mask_path)

if __name__ == "__main__":
    generate_all_differences()
    print("\nDiferencias visuales guardadas en data/results/")
