"""Three-frame empty-left-slot classifier for policy routing."""
import numpy as np
import torch
from torch import nn
from torchvision import models

from .learning import image_tensor, load_checkpoint


class SlotClassifier(nn.Module):
    def __init__(self, encoder=None):
        super(SlotClassifier, self).__init__()
        if encoder is None:
            try:
                encoder = models.mobilenet_v2(weights=None)
            except TypeError:  # torchvision shipped with JetPack / PyTorch 1.6
                encoder = models.mobilenet_v2(pretrained=False)
            encoder.classifier = nn.Identity()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Linear(1280, 128), nn.ReLU(), nn.Dropout(.25), nn.Linear(128, 1))

    def forward(self, images):
        batch, frames = images.shape[:2]
        features = self.encoder(images.reshape(-1, *images.shape[2:]))
        features = features.reshape(batch, frames, -1).mean(dim=1)
        return self.head(features).squeeze(-1)

    def train(self, mode=True):
        super(SlotClassifier, self).train(mode)
        self.encoder.eval()
        return self


class SlotPredictor:
    def __init__(self, checkpoint, device="cpu"):
        saved = load_checkpoint(checkpoint, device)
        if saved.get("architecture") != "mobilenet-v2-mean3-mlp128":
            raise ValueError("Unsupported slot-classifier architecture")
        self.model = SlotClassifier().to(device)
        self.model.load_state_dict(saved["state_dict"])
        self.model.eval()
        self.device = torch.device(device)
        self.threshold = float(saved.get("threshold", .5))

    def predict(self, three_rgb_frames):
        if len(three_rgb_frames) != 3:
            raise ValueError("Exactly three starting frames are required")
        tensors = [image_tensor(np.asarray(frame)) for frame in three_rgb_frames]
        images = torch.stack(tensors).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probability = float(torch.sigmoid(self.model(images))[0].item())
        return {"left_probability": probability,
                "left_empty": probability >= self.threshold,
                "threshold": self.threshold}


def classify_start(camera, predictor, frame_callback=None):
    """Read three stationary frames and return the route decision."""
    frames = []
    last_frame_id = -1
    for index in range(3):
        frame = camera.read(after=last_frame_id, timeout=2.)
        last_frame_id = frame.get("frame_id", last_frame_id)
        frames.append(frame["rgb"])
        if frame_callback is not None:
            frame_callback(frame, index + 1)
    return predictor.predict(frames)
