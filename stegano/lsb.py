# Copyright (c) Universidad Politecnica Madrid, 2025
# Authors: Alejandro Náger Fernández-Calvo <a.nager@alumnos.upm.es>
# Dates:
#  Creation: Mar. 24, 2025
#  Modification: Apr. 17, 2025
# Documented by: Alejandro Náger Fernández-Calvo <a.nager@alumnos.upm.es>

# stegano/lsb.py

from PIL import Image
from stegano.utils import text_to_bits, bits_to_text, add_delimiter, remove_delimiter

def encode_lsb(image_path: str, message: str, output_path: str, delimiter: str = "#####") -> None:
    """Oculta un mensaje en la imagen utilizando LSB y guarda el resultado."""
    # 1. Cargar y convertir a RGB
    image = Image.open(image_path).convert("RGB")
    pixels = list(image.getdata())

    # 2. Transformar mensaje a bits y añadir delimitador
    binary_message = text_to_bits(add_delimiter(message, delimiter))
    message_len = len(binary_message)

    # 3. Comprobar capacidad: 3 bits por pixel
    if message_len > len(pixels) * 3:
        raise ValueError("El mensaje es demasiado largo para la capacidad de la imagen.")

    # 4. Inserción bit a bit de los LSB de R, G y B
    new_pixels = []
    bit_index = 0
    for r, g, b in pixels:
        if bit_index < message_len:
            r = (r & ~1) | int(binary_message[bit_index])
            bit_index += 1
        if bit_index < message_len:
            g = (g & ~1) | int(binary_message[bit_index])
            bit_index += 1
        if bit_index < message_len:
            b = (b & ~1) | int(binary_message[bit_index])
            bit_index += 1
        new_pixels.append((r, g, b))

    # 5. Guardar imagen estego
    image.putdata(new_pixels)
    image.save(output_path)

def decode_lsb(image_path: str, delimiter: str = "#####") -> str:
    """Extrae un mensaje oculto de la imagen utilizando LSB."""
    image = Image.open(image_path)
    image = image.convert("RGB")
    pixels = list(image.getdata())

    # 1. Leer los LSB de cada canal
    bits = ""
    for r, g, b in pixels:
        bits += str(r & 1) + str(g & 1) + str(b & 1)

    # 2. Reconstruir texto y eliminar delimitador
    message = bits_to_text(bits)
    return remove_delimiter(message, delimiter)