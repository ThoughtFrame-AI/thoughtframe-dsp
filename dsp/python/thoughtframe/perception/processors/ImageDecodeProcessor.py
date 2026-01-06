# thoughtframe/perception/processors/ImageDecodeProcessor.py

import io
import requests
import numpy as np
from PIL import Image

from thoughtframe.perception.interface import PerceptionProcessor, PerceptionAnalysis


class ImageDecodeProcessor(PerceptionProcessor):
    OP_NAME = "image_decode"

    def process(self, analysis: PerceptionAnalysis) -> None:
        # Tensor already exists (synthetic, video, etc.)
        if analysis.tensor is not None:
            return

        item = analysis.item
        url = item.get("url")
        if not url:
            raise RuntimeError("image_decode requires 'url' or preexisting tensor")

        resp = requests.get(url, timeout=10)
        resp.raise_for_status()

        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        arr = np.asarray(img)

        analysis.tensor = arr
        analysis.metadata["decoded_shape"] = arr.shape
        analysis.metadata["decoded_dtype"] = str(arr.dtype)

    @classmethod
    def from_config(cls, cfg, source):
        return cls()
