import io
from PIL import Image
import numpy as np


def decode_image(image_bytes: bytes) -> tuple[np.ndarray, dict]:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    width, height = image.size
    arr = np.array(image)
    meta = {"width": width, "height": height}
    return arr, meta