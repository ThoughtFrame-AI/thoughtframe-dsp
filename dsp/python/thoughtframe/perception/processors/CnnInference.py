# thoughtframe/perception/processors/CnnInferenceProcessor.py

import numpy as np
from thoughtframe.perception.interface import PerceptionProcessor, PerceptionAnalysis


class CnnInference(PerceptionProcessor):
    OP_NAME = "cnn_inference"

    def __init__(self, model, device="cpu"):
        self.model = model
        self.device = device

    def process(self, analysis: PerceptionAnalysis) -> None:
        if analysis.tensor is None:
            return

        import torch

        # numpy → torch
        x = torch.from_numpy(analysis.tensor).unsqueeze(0)
        x = x.to(self.device)

        with torch.no_grad():
            logits = self.model(x)

        # store raw output; downstream decides meaning
        analysis.metadata["logits"] = logits.cpu().numpy()

    @classmethod
    def from_config(cls, cfg, source):
        framework = cfg.get("framework", "torchvision")
        model_name = cfg.get("model", "resnet50")
        weights = cfg.get("weights", "imagenet")
        device = cfg.get("device", "cpu")

        if framework != "torchvision":
            raise ValueError(f"Unsupported framework: {framework}")

        import torch
        import torchvision.models as models

        # Load model + weights
        if model_name == "resnet50":
            if weights == "imagenet":
                model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            else:
                model = models.resnet50()
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        model.eval()
        model.to(device)

        return cls(model=model, device=device)
