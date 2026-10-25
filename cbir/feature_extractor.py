"""VGG16 deep feature extractor (the professor's FeatureExtractor, plus batching).

Same pipeline as the course notebook and the reference project: ImageNet VGG16, output of the
`fc1` layer (4096 values), image resized to 224x224 RGB, `preprocess_input`, L2-normalised.
"""
import os

import numpy as np
from PIL import Image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image

# See https://keras.io/api/applications/ for details


class FeatureExtractor:
    def __init__(self, weights=None):
        # CBIR_VGG_WEIGHTS=none builds an untrained network: only for tests without internet.
        weights = weights or os.environ.get("CBIR_VGG_WEIGHTS", "imagenet")
        base_model = VGG16(weights=None if str(weights).lower() == "none" else weights)
        self.model = Model(inputs=base_model.input, outputs=base_model.get_layer("fc1").output)

    @staticmethod
    def _to_array(img):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        img = img.resize((224, 224))      # VGG must take a 224x224 img as an input
        img = img.convert("RGB")          # Make sure img is color
        return image.img_to_array(img)    # Height x Width x Channel, float32

    def extract(self, img):
        """Deep feature of one PIL image / RGB array: np.ndarray with shape (4096,), unit norm."""
        x = np.expand_dims(self._to_array(img), axis=0)   # (H, W, C) -> (1, H, W, C)
        x = preprocess_input(x)                           # subtract the ImageNet channel means
        feature = self.model.predict(x, verbose=0)[0]     # (1, 4096) -> (4096,)
        return feature / np.linalg.norm(feature)

    def extract_batch(self, imgs, batch_size=32):
        """Same as `extract` for a list of images, several per forward pass (much faster)."""
        out = []
        for s in range(0, len(imgs), batch_size):
            x = np.stack([self._to_array(im) for im in imgs[s:s + batch_size]])
            f = self.model.predict(preprocess_input(x), verbose=0)
            out.append(f / np.linalg.norm(f, axis=1, keepdims=True))
        return np.concatenate(out) if out else np.zeros((0, 4096), np.float32)
