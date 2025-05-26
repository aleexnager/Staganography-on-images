# Copyright (c) Universidad Politecnica Madrid, 2025
# Authors: Alejandro Náger Fernández-Calvo <a.nager@alumnos.upm.es>
# Dates:
#  Creation: Mar. 24, 2025
#  Modification: Apr. 17, 2025
# Documented by: Alejandro Náger Fernández-Calvo <a.nager@alumnos.upm.es>

#tests/test_lsb.py

import os
import pytest
import numpy as np
from stegano.lsb import encode_lsb, decode_lsb
from stegano.utils import load_text_file
from stegano.metrics import calculate_psnr, calculate_ssim, calculate_capacity, generate_heatmap, calculate_pixels_modified_ratio, calculate_bits_modified_ratio
from PIL import Image

# Rutas
INPUT_IMAGE_DIR = "data/input"
MESSAGE_DIR = "data/messages"
OUTPUT_IMAGE_DIR = "data/output"
RESULTS_DIR = "data/results"

os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

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

# Resultados
@pytest.fixture(scope="session", autouse=True)
def share_results(request):
    return request.config.results

# Luego, dentro del test usa esta forma:
def test_lsb_pipeline(image_name, message_name, share_results):
    ...
    share_results.append({...})

@pytest.mark.parametrize("image_name, message_name", test_cases)
def test_lsb_pipeline(image_name, message_name, share_results):
    input_image_path = os.path.join(INPUT_IMAGE_DIR, image_name)
    message_path = os.path.join(MESSAGE_DIR, message_name)
    output_image_path = os.path.join(OUTPUT_IMAGE_DIR, f"lsb_{image_name}")

    message = load_text_file(message_path)
    encode_lsb(input_image_path, message, output_image_path)

    extracted_message = decode_lsb(output_image_path)
    assert message == extracted_message, "El mensaje extraído no coincide con el original"

    original_img = Image.open(input_image_path).convert("RGB")
    stego_img = Image.open(output_image_path).convert("RGB")

    psnr = calculate_psnr(original_img, stego_img)
    ssim = calculate_ssim(original_img, stego_img)
    capacity = calculate_capacity(message, stego_img)

    w, h = original_img.size
    slots_map = np.full((h, w), 3, dtype=int)

    pixels_ratio = calculate_pixels_modified_ratio(message, slots_map)
    bit_ratio = calculate_bits_modified_ratio(original_img, stego_img)

    # Guardar métricas
    share_results.append({
        "image": image_name,
        "message": message_name,
        "psnr": psnr,
        "ssim": ssim,
        "capacity": capacity,
        "pixels_modified_ratio": pixels_ratio,
        "bits_modified_ratio": bit_ratio,
        "length": len(message),
        "method": "lsb"
    })

    heatmap_path = os.path.join(
        "data", "heatmaps",
        f"heatmap_lsb_{image_name.replace('.png','')}_{message_name.replace('.txt','')}.png"
    )
    generate_heatmap(
        os.path.join(INPUT_IMAGE_DIR, image_name),
        output_image_path,
        output_path=heatmap_path,
        title=f"Heatmap LSB: {image_name} / {message_name}"
    )
