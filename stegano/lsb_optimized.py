# Copyright (c) Universidad Politecnica Madrid, 2025
# Authors: Alejandro Náger Fernández-Calvo <a.nager@alumnos.upm.es>
# Dates:
#  Creation: Mar. 24, 2025
#  Modification: Apr. 17, 2025
# Documented by: Alejandro Náger Fernández-Calvo <a.nager@alumnos.upm.es>

# stegano/lsb_optimized.py

import os
from PIL import Image
import numpy as np
from stegano.utils import text_to_bits, bits_to_text, add_delimiter, remove_delimiter
from stegano.entropy import compute_entropy_map, normalize_entropy_map

def encode_lsb_optimized(image_path: str,
                         message: str,
                         output_path: str,
                         window_size: int = 3,
                         threshold: float = 0.5,
                         delimiter: str = "#####") -> None:
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img)
    # Mapa de entropía normalizado
    ent_norm = normalize_entropy_map(compute_entropy_map(img, window_size))
    mask_positions = np.argwhere(ent_norm >= threshold)
    if mask_positions.size == 0:
        raise ValueError("No se encontraron regiones con entropía suficiente.")
    bits = text_to_bits(add_delimiter(message, delimiter))
    total_bits = len(bits)
    if total_bits > len(mask_positions) * 3:
        raise ValueError("El mensaje es demasiado largo para la capacidad disponible.")
    stego = arr.copy()
    bit_idx = 0
    for y, x in mask_positions:
        if bit_idx >= total_bits: break
        for c in range(3):
            if bit_idx < total_bits:
                stego[y, x, c] = (stego[y, x, c] & 0xFE) | int(bits[bit_idx])
                bit_idx += 1
    Image.fromarray(stego).save(output_path)


def decode_lsb_optimized(image_path: str,
                         window_size: int = 3,
                         threshold: float = 0.5,
                         delimiter: str = "#####") -> str:
    """
    Extrae un mensaje oculto usando LSB en píxeles seleccionados según entropía local
    calculada sobre la imagen original.
    """
    # Cargar imagen stego y su array
    stego_img = Image.open(image_path).convert("RGB")
    stego_arr = np.array(stego_img)

    # Derivar ruta de la imagen original a partir del nombre de archivo
    base = os.path.basename(image_path)
    if base.startswith("lsbopt_"):
        orig_name = base[len("lsbopt_"):]
    elif base.startswith("lsb_"):
        orig_name = base[len("lsb_"):]
    else:
        orig_name = base
    orig_path = os.path.join("data", "input", orig_name)
    orig_img = Image.open(orig_path).convert("RGB")

    # Calcular el mismo mapa de entropía sobre la original
    ent_norm = normalize_entropy_map(compute_entropy_map(orig_img, window_size))
    mask_positions = np.argwhere(ent_norm >= threshold)

    # Extraer bit a bit, con parada al encontrar el delimitador
    bits = ""
    message = ""
    for y, x in mask_positions:
        for c in range(3):
            bits += str(int(stego_arr[y, x, c]) & 1)
            if len(bits) >= 8:
                byte = bits[:8]
                bits = bits[8:]
                char = chr(int(byte, 2))
                message += char
                if message.endswith(delimiter):
                    return message[:-len(delimiter)]

    # Si nunca vimos el delimitador, devolvemos todo lo leído
    return message
