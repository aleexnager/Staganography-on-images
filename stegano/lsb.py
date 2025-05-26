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
    image = Image.open(image_path)
    image = image.convert("RGB")
    pixels = list(image.getdata())

    binary_message = text_to_bits(add_delimiter(message, delimiter))
    message_len = len(binary_message)

    if message_len > len(pixels) * 3:
        raise ValueError("El mensaje es demasiado largo para esta imagen.")

    new_pixels = []
    bit_index = 0

    for pixel in pixels:
        r, g, b = pixel
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

    image.putdata(new_pixels)
    image.save(output_path)

def decode_lsb(image_path: str, delimiter: str = "#####") -> str:
    """Extrae un mensaje oculto de la imagen utilizando LSB."""
    image = Image.open(image_path)
    image = image.convert("RGB")
    pixels = list(image.getdata())

    bits = ""
    for pixel in pixels:
        for channel in pixel[:3]:
            bits += str(channel & 1)

    message = bits_to_text(bits)
    return remove_delimiter(message, delimiter)
