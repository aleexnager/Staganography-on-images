# stegano/metrics.py

import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from stegano.utils import text_to_bits

def calculate_psnr(original: Image.Image, stego: Image.Image) -> float:
    """Calcula el PSNR entre dos imágenes."""
    original_np = np.array(original)
    stego_np = np.array(stego)
    return peak_signal_noise_ratio(original_np, stego_np, data_range=255)

def calculate_ssim(original: Image.Image, stego: Image.Image) -> float:
    """Calcula el SSIM entre dos imágenes RGB."""
    original_np = np.array(original)
    stego_np = np.array(stego)
    return structural_similarity(original_np, stego_np, channel_axis=-1)

def calculate_capacity(message: str, image: Image.Image) -> float:
    """Calcula la capacidad de ocultación en bits por píxel (bpp)."""
    total_bits = len(text_to_bits(message))
    width, height = image.size
    total_pixels = width * height
    return total_bits / total_pixels

def generate_heatmap(original_path, stego_path, output_path=None, title=None):
    original = np.array(Image.open(original_path).convert("RGB"), dtype=np.int16)
    stego = np.array(Image.open(stego_path).convert("RGB"), dtype=np.int16)

    diff = np.abs(original - stego)  # diferencias por canal
    diff_gray = np.mean(diff, axis=2)  # diferencia media por píxel (opcional)

    plt.figure(figsize=(8, 6))
    plt.imshow(diff_gray, cmap='hot', interpolation='nearest')
    plt.axis('off')
    if title:
        plt.title(title)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def calculate_pixels_modified_ratio(message, slots_map):
    """
    Calcula la proporción de píxeles modificados dados los bits disponibles por píxel en slots_map.
    """
    total_bits = len(message) * 8  # bits a ocultar
    total_capacity = slots_map.sum()

    if total_capacity < total_bits:
        raise ValueError("No hay suficiente capacidad para ocultar el mensaje")

    # N° de píxeles modificados = ceil(total_bits / bits_por_píxel)
    modified_pixels = 0
    bits_remaining = total_bits

    # Plano por plano (prioriza zonas con más bits disponibles)
    for bits in [3, 2, 1]:
        count = np.sum(slots_map == bits)
        usable_bits = count * bits
        if bits_remaining <= usable_bits:
            modified_pixels += (bits_remaining + bits - 1) // bits  # ceil
            bits_remaining = 0
            break
        else:
            modified_pixels += count
            bits_remaining -= usable_bits

    h, w = slots_map.shape
    return modified_pixels / (h * w)

def calculate_bits_modified_ratio(original: Image.Image, stego: Image.Image) -> float:
    """
    Calcula la fracción de bits modificados entre dos imágenes RGB.
    """
    # Convertir a array uint8
    orig_np = np.array(original, dtype=np.uint8)
    stego_np = np.array(stego, dtype=np.uint8)

    # XOR para obtener bits cambiados en cada canal
    diff = np.bitwise_xor(orig_np, stego_np)  # shape (H, W, 3)

    # Convertir cada canal a bits y contar 1s
    # unpackbits sobre el último eje
    bits = np.unpackbits(diff, axis=-1)  # shape (H, W, 3, 8)
    total_bits = bits.size
    modified_bits = bits.sum()

    return modified_bits / total_bits
