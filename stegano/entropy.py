# Copyright (c) Universidad Politecnica Madrid, 2025
# Authors: Alejandro Náger Fernández-Calvo <a.nager@alumnos.upm.es>
# Dates:
#  Creation: Mar. 24, 2025
#  Modification: Apr. 17, 2025
# Documented by: Alejandro Náger Fernández-Calvo <a.nager@alumnos.upm.es>

# stegano/entropy.py

from skimage.filters.rank import entropy as sk_entropy
from skimage.morphology import square
import numpy as np
from PIL import Image


def compute_entropy_map(image: Image.Image, window_size: int = 3) -> np.ndarray:
    """
    Calcula el mapa de entropía local de una imagen en escala de grises.

    :param image: Imagen PIL a color o escala de grises.
    :param window_size: Tamaño de la ventana cuadrada para calcular la entropía.
    :return: Mapa de entropía como arreglo numpy de tipo float (bits).
    """
    # Convertir a escala de grises y a matriz uint8
    gray = image.convert("L")
    arr = np.array(gray)

    # Calcular entropía local con skimage (devuelve valores de 0 a log2(window_size^2))
    foot = square(window_size)
    ent = sk_entropy(arr, footprint=foot)

    # Convertir a bits: Skimage rank.entropy devuelve entropía en bits
    return ent.astype(np.float32)


def normalize_entropy_map(entropy_map: np.ndarray) -> np.ndarray:
    """
    Normaliza un mapa de entropía al rango [0,1].

    :param entropy_map: Mapa de entropía.
    :return: Mapa normalizado.
    """
    min_val = np.min(entropy_map)
    max_val = np.max(entropy_map)
    if max_val - min_val == 0:
        return np.zeros_like(entropy_map)
    return (entropy_map - min_val) / (max_val - min_val)
