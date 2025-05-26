# tests/test_lsb_adaptive.py

import os
import pytest
import numpy as np
from PIL import Image

from stegano.entropy import compute_entropy_map, normalize_entropy_map
from stegano.lsb_adaptive import encode_lsb_adaptive, decode_lsb_adaptive
from stegano.utils import load_text_file
from stegano.metrics import (
    calculate_psnr,
    calculate_ssim,
    calculate_capacity,
    calculate_pixels_modified_ratio,
    calculate_bits_modified_ratio
)

INPUT_DIR = "data/input"
MESSAGE_DIR = "data/messages"
OUTPUT_DIR = "data/output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Casos de prueba
test_cases = [
    ("pixel_flowers.png", "msg1.txt"),
    ("pixel_flowers.png", "msg2.txt"),
    ("pixel_flowers.png", "msg3.txt"),
    ("yak.png", "msg1.txt"),
    ("yak.png", "msg2.txt"),
    ("yak.png", "msg3.txt"),
    ("snake.png", "msg1.txt"),
    ("snake.png", "msg2.txt"),
    ("snake.png", "msg3.txt"),
    ("yellow_flowers.png", "msg1.txt"),
    ("yellow_flowers.png", "msg2.txt"),
    ("yellow_flowers.png", "msg3.txt")
]

@pytest.fixture(scope="session", autouse=True)
def share_results(request):
    return request.config.results

@pytest.mark.parametrize("image_name,message_name", test_cases)
def test_lsb_adaptive_pipeline(image_name, message_name, share_results):
    inp = os.path.join(INPUT_DIR, image_name)
    msgp = os.path.join(MESSAGE_DIR, message_name)
    out = os.path.join(OUTPUT_DIR, f"lsbadapt_{image_name}")

    # Leer y ocultar
    message = load_text_file(msgp)
    encode_lsb_adaptive(
        inp,
        message,
        out,
        window_size=3,
        thresholds=(0.5, 0.75),
        bits_per_channel=(0, 1, 2),
        delimiter="#####"
    )

    # Extraer
    extracted = decode_lsb_adaptive(
        out,
        inp,
        window_size=3,
        thresholds=(0.5, 0.75),
        bits_per_channel=(0, 1, 2),
        delimiter="#####"
    )
    assert extracted == message, "El mensaje extraído no coincide"

    # Cargar para métricas
    original = Image.open(inp).convert("RGB")
    stego    = Image.open(out).convert("RGB")

    psnr     = calculate_psnr(original, stego)
    ssim     = calculate_ssim(original, stego)
    capacity = calculate_capacity(message, stego)

    # Reconstruir slot_map de bits por canal
    ent_norm = normalize_entropy_map(compute_entropy_map(original))
    t0, t1 = (0.5, 0.75)
    b0, b1, b2 = (0, 1, 2)
    h, w = original.height, original.width

    chan_slots = np.zeros((h, w), dtype=int)
    chan_slots[ent_norm < t0] = b0
    chan_slots[(ent_norm >= t0) & (ent_norm < t1)] = b1
    chan_slots[ent_norm >= t1] = b2

    # Total bits disponibles por píxel = slots_canal * 3 canales
    bits_per_pixel_map = chan_slots * 3

    # Métrica de píxeles modificados
    pixels_ratio = calculate_pixels_modified_ratio(message, bits_per_pixel_map)
    # Métrica de bits modificados
    bit_ratio    = calculate_bits_modified_ratio(original, stego)

    # Guardar resultados
    share_results.append({
        "image": image_name,
        "message": message_name,
        "psnr": psnr,
        "ssim": ssim,
        "capacity": capacity,
        "pixels_modified_ratio": pixels_ratio,
        "bits_modified_ratio": bit_ratio,
        "length": len(message),
        "method": "lsbadapt"
    })
