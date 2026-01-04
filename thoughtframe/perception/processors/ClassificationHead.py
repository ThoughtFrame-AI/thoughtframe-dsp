# thoughtframe/perception/processors/ClassificationHead.py

import numpy as np
from thoughtframe.perception.interface import PerceptionProcessor, PerceptionAnalysis


class ClassificationHead(PerceptionProcessor):
    OP_NAME = "classification_head"

    def __init__(self, labels=None, top_k=5):
        self.labels = labels
        self.top_k = top_k

    def process(self, analysis: PerceptionAnalysis) -> None:
        logits = analysis.metadata.get("logits")
        if logits is None:
            return

        # Flatten logits to 1D
        logits = np.asarray(logits).squeeze()

        # Softmax (numerically stable)
        exps = np.exp(logits - logits.max())
        probs = exps / exps.sum()

        # Top-k indices
        topk_idx = np.argsort(probs)[-self.top_k:][::-1]

        results = []
        for i in topk_idx:
            label = self.labels[i] if self.labels else str(i)
            results.append({
                "class_id": int(i),
                "label": label,
                "probability": float(probs[i]),
            })
        
        analysis.metadata["top_k"] = results
        print(results)
        
      
    @classmethod
    def from_config(cls, cfg, source):
        top_k = cfg.get("top_k", 5)

        labels = None
        label_spec = cfg.get("labels")

        if label_spec == "imagenet":
            import torchvision.models as models
            labels = models.ResNet50_Weights.IMAGENET1K_V1.meta["categories"]

        return cls(labels=labels, top_k=top_k)


    
   
