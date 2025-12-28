import numpy as np
from thoughtframe.bootstrap import thoughtframe
from thoughtframe.sensor.interface import (
    AcousticChunkProcessor,
    AcousticAnalysis,
)


class GuardProcessor(AcousticChunkProcessor):
 
    def __init__(self, cfg: dict, sensor, children: list[AcousticChunkProcessor]):
        self.cfg = cfg
        self.sensor = sensor
        self.name = cfg.get("name", "guard")

        self.lpf = cfg.get("lpf")
        self.hpf = cfg.get("hpf")
        self.band = cfg.get("band")

        # Child processors (already instantiated)
        self.children = children

        # Optional filter state (for future IIR/FIR use)
        self._filter_state = None

   
    @classmethod
    def from_config(cls, cfg, sensor):
        """
        Create GuardProcessor and its child processors
        using the same construction authority as the mesh.
        """
        mesh = thoughtframe.get("sensormeshmanager")

        children = []
        for child_cfg in cfg.get("pipeline", []):
            child = mesh.processor_manager.createProcessor(child_cfg, sensor)
            children.append(child)

        return cls(cfg, sensor, children)

 
    def _apply_filter(self, x: np.ndarray, analysis) -> np.ndarray:
        fs = self.sensor.fs
        n = x.shape[0]
    
        # FFT
        X = analysis.fft
        freqs = analysis.fft_freqs
    
        mask = np.ones_like(X, dtype=np.float32)
    
        # Band-pass
        if self.band:
            low, high = self.band
            mask = (freqs >= low) & (freqs <= high)
    
        # High-pass
        elif self.hpf:
            mask = freqs >= self.hpf
    
        # Low-pass
        elif self.lpf:
            mask = freqs <= self.lpf
    
        # Apply mask
        X_filtered = X * mask
    
        # Inverse FFT
        y = np.fft.irfft(X_filtered, n=n)
    
        return y.astype(x.dtype, copy=False)

    
    def process(self, chunk: np.ndarray, analysis: AcousticAnalysis) -> None:
        if chunk.size == 0:
            return
        cloned = chunk.copy()
        cloned = self._apply_filter(cloned, analysis)
        for child in self.children:
            child.process(cloned, analysis)

        

   

    
    


        

   
