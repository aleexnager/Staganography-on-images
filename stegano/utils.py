# Copyright (c) Universidad Politecnica Madrid, 2025
# Authors: Alejandro Náger Fernández-Calvo <a.nager@alumnos.upm.es>
# Dates:
#  Creation: Mar. 24, 2025
#  Modification: Apr. 17, 2025
# Documented by: Alejandro Náger Fernández-Calvo <a.nager@alumnos.upm.es>

# stegano/utils.py

def text_to_bits(text: str) -> str:
    """Convierte una cadena de texto a una cadena binaria."""
    return ''.join(format(ord(char), '08b') for char in text)

def bits_to_text(bits: str) -> str:
    """Convierte una cadena binaria a texto."""
    chars = [bits[i:i+8] for i in range(0, len(bits), 8)]
    return ''.join(chr(int(b, 2)) for b in chars)

def add_delimiter(message: str, delimiter: str = "#####") -> str:
    """Añade un delimitador al final del mensaje para indicar su final."""
    return message + delimiter

def remove_delimiter(message: str, delimiter: str = "#####") -> str:
    """Elimina el delimitador del mensaje extraído."""
    return message.split(delimiter)[0]

def load_text_file(path: str) -> str:
    """Lee un archivo de texto y devuelve su contenido."""
    with open(path, "r", encoding="utf-8") as file:
        return file.read()
