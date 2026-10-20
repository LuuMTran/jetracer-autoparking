"""Mild camera variation, sampled once per recurrent training window."""
import math
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


DEFAULT_AUGMENTATION = {
    "probability": .75, "rotation_degrees": 3., "brightness": .20,
    "contrast": .15, "saturation": .15, "noise_std": .012,
    "blur_probability": .15, "blur_radius": .6,
}


class SequenceAugment:
    """Share geometry and lighting across frames; sensor noise is independent.

    Rotation simulates small camera roll, not a different vehicle heading.
    No flips, large crops or translations that would invalidate action labels.
    """
    def __init__(self, config=None, seed=7):
        self.config = dict(DEFAULT_AUGMENTATION if config is None else config)
        if set(self.config) != set(DEFAULT_AUGMENTATION):
            raise ValueError("Unexpected augmentation settings")
        if any(not math.isfinite(float(v)) or float(v) < 0 for v in self.config.values()):
            raise ValueError("Augmentation values must be finite and nonnegative")
        for name in ("probability", "blur_probability", "brightness", "contrast", "saturation"):
            if self.config[name] > 1:
                raise ValueError(name + " must be at most 1")
        self.rng = np.random.RandomState(seed)

    def sample(self):
        c, r = self.config, self.rng
        if r.rand() >= c["probability"]:
            return None
        return {"angle": r.uniform(-c["rotation_degrees"], c["rotation_degrees"]),
                "brightness": r.uniform(1 - c["brightness"], 1 + c["brightness"]),
                "contrast": r.uniform(1 - c["contrast"], 1 + c["contrast"]),
                "saturation": r.uniform(1 - c["saturation"], 1 + c["saturation"]),
                "noise_std": r.uniform(0, c["noise_std"]),
                "blur_radius": r.uniform(0, c["blur_radius"]) if r.rand() < c["blur_probability"] else 0.}

    def apply(self, image, parameters):
        image = image.convert("RGB").resize((224, 224), Image.BILINEAR)
        if parameters is None:
            return image
        p = parameters
        if p["angle"]:
            # Reflect the border before rotating, then retain the original field
            # of view. This avoids learning artificial black corner triangles.
            padded = np.pad(np.asarray(image), ((32, 32), (32, 32), (0, 0)), mode="reflect")
            image = Image.fromarray(padded).rotate(p["angle"], resample=Image.BILINEAR)
            image = image.crop((32, 32, 256, 256))
        image = ImageEnhance.Brightness(image).enhance(p["brightness"])
        image = ImageEnhance.Contrast(image).enhance(p["contrast"])
        image = ImageEnhance.Color(image).enhance(p["saturation"])
        if p["blur_radius"]:
            image = image.filter(ImageFilter.GaussianBlur(p["blur_radius"]))
        if p["noise_std"]:
            pixels = np.asarray(image, dtype=np.float32) / 255.
            pixels += self.rng.normal(0, p["noise_std"], pixels.shape).astype(np.float32)
            image = Image.fromarray(np.uint8(np.clip(pixels, 0, 1) * 255. + .5))
        return image
