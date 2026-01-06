# thoughtframe/perception/processors/DebugProbeProcessor.py

from thoughtframe.perception.interface import PerceptionProcessor, PerceptionAnalysis
import numpy as np


class TensorContract(PerceptionProcessor):
    OP_NAME = "tensor_contract"

    def process(self, analysis: PerceptionAnalysis) -> None:
        t = analysis.tensor
    
        # HWC → CHW
        if t.ndim == 3 and t.shape[-1] in (1, 3):
            t = np.transpose(t, (2, 0, 1))
    
        # uint8 → float32
        t = t.astype(np.float32, copy=False)
    
        # scale to [0,1]
        t /= 255.0
    
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
        t = (t - mean) / std
    
        analysis.tensor = t

    @classmethod
    def from_config(cls, cfg, source):
        return cls()
