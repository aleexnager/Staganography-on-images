# Copyright (c) Universidad Politecnica Madrid, 2025
# Authors: Alejandro Náger Fernández-Calvo <a.nager@alumnos.upm.es>
# Dates:
#  Creation: Apr. 3, 2025
#  Modification: Apr. 17, 2025
# Documented by: Alejandro Náger Fernández-Calvo <a.nager@alumnos.upm.es>

# stegano/lsb_adaptive.py

import os
from PIL import Image
import numpy as np
from stegano.utils import text_to_bits, add_delimiter
from stegano.entropy import compute_entropy_map, normalize_entropy_map

def encode_lsb_adaptive(image_path: str,
                        message: str,
                        output_path: str,
                        window_size: int = 3,
                        thresholds: tuple[float,float] = (0.5, 0.75),
                        bits_per_channel: tuple[int,int,int] = (0, 1, 2),
                        delimiter: str = "#####") -> None:
    """
    Multi‐LSB adaptativo por canal:
      entropy < t0      → 0 bits/canal
      t0 ≤ entropy < t1 → 1 bit/canal
      entropy ≥ t1      → 2 bits/canal
    """

    # 1. Cargar imagen
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img)
    h, w, _ = arr.shape

    # 2. Calc. y normalizar mapa de entropía local sobre imagen original
    ent_norm = normalize_entropy_map(compute_entropy_map(img, window_size))
    
    # 3. Extraer los umbrales y configuración de bits por canal
    t0, t1 = thresholds
    b0, b1, b2 = bits_per_channel

    # 4. Generar mapa de bits por canal
    chan_slots = np.zeros((h, w), dtype=int)
    chan_slots[ent_norm < t0] = b0
    chan_slots[(ent_norm >= t0) & (ent_norm < t1)] = b1
    chan_slots[ent_norm >= t1] = b2

    # 5. Preparar el mensaje para su inserción
    bits = text_to_bits(add_delimiter(message, delimiter))
    total_bits = len(bits)
    capacity = int(np.sum(chan_slots) * 3) # Capacidad total en bits (3 bits por canal)

    # 6. Comprobar capacidad
    if total_bits > capacity:
        raise ValueError("El mensaje es demasiado largo para la capacidad de la imagen.")

    # 7. Crear copia de la imagen para modificar
    stego = arr.copy()
    bit_idx = 0

    # 8. Inserción de bits canal por canal
    ys, xs = np.nonzero(chan_slots > 0)
    positions = sorted(zip(ys, xs), key=lambda p: chan_slots[p[0], p[1]], reverse=True)

    for y, x in positions:
        bpc = chan_slots[y, x]   # Bits por canal en este píxel
        for channel in range(3): # Iterar sobre canales R, G, B
            for k in range(bpc): # Insertar bits menos significativos
                if bit_idx >= total_bits:
                    break
                # Limpiza el bit k
                mask = (~(1 << k)) & 0xFF
                orig_val = int(stego[y, x, channel])
                # Construir nuevo bit
                new_bit = (int(bits[bit_idx]) & 1) << k
                # Sustituir bit k
                stego[y, x, channel] = (orig_val & mask) | new_bit
                bit_idx += 1
            if bit_idx >= total_bits:
                break
        if bit_idx >= total_bits:
            break

    # 9. Guardar imagen estego
    Image.fromarray(stego.astype(np.uint8)).save(output_path)


def decode_lsb_adaptive(image_path: str,
                        original_path: str,
                        window_size: int = 3,
                        thresholds: tuple[float,float] = (0.5, 0.75),
                        bits_per_channel: tuple[int,int,int] = (0, 1, 2),
                        delimiter: str = "#####") -> str:
    """
    Decodifica el multi‐LSB adaptativo.
    """

    # 1. Cargar imagenes estego y original
    stego_img = Image.open(image_path).convert("RGB")
    stego = np.array(stego_img)
    orig_img = Image.open(original_path).convert("RGB")

    # 2. Calc. y normalizar mapa de entropía local sobre imagen original
    ent_norm = normalize_entropy_map(compute_entropy_map(orig_img, window_size))
    # 3. Extraer los umbrales y configuración de bits por canal
    t0, t1 = thresholds
    b0, b1, b2 = bits_per_channel

    # 4. Mapa indicando n bits por canal extraer en cada píxel
    h, w, _ = stego.shape
    chan_slots = np.zeros((h, w), dtype=int)
    
    # 5. Asignar bits por canal según el valor de entropía local
    chan_slots[ent_norm < t0] = b0
    chan_slots[(ent_norm >= t0) & (ent_norm < t1)] = b1
    chan_slots[ent_norm >= t1] = b2

    # 6. Coordenadas de píxeles con al menos 1 bit disponible
    ys, xs = np.nonzero(chan_slots > 0)

    # 7. Ordenar las posiciones por cantidad de bits disponibles (de mayor a menor)
    positions = sorted(zip(ys, xs), key=lambda p: chan_slots[p[0], p[1]], reverse=True)

    # 8. Extraer bits de la imagen estego
    bits = ""
    message = ""
    for y, x in positions:
        bpc = chan_slots[y, x]   # Bits por canal en este píxel
        for channel in range(3): # Iterar sobre canales R, G, B
            for k in range(bpc): # Extraer bits menos significativos
                bits += str((stego[y, x, channel] >> k) & 1)
                # Cuando se acumulan 8 bits, convertirlos a un carácter
                if len(bits) >= 8:
                    byte, bits = bits[:8], bits[8:]
                    char = chr(int(byte, 2))
                    message += char
                    if message.endswith(delimiter):
                        return message[:-len(delimiter)]
    # 9. Devolver mensaje completo si no se encontró el delimitador
    return message
