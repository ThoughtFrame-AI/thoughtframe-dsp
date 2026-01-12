import torch
import numpy as np
from collections import deque
from thoughtframe.models.dcmt_model import DcmtModel
from thoughtframe.sensor.interface import AcousticChunkProcessor

class ComplexDcmtEmbeddingProcessor(AcousticChunkProcessor):
    """
    Complex-Valued DCMT Processor.
    
    Instead of LogMel (Magnitude only), this consumes the raw Complex STFT.
    It applies Phase-Preserving Dynamic Range Compression.
    It feeds a 2-channel (Real, Imag) tensor to the network.
    
    Theory:
    - Background Noise (Phase Incoherent) -> Destructive Interference -> Low Norm
    - Signal/Structure (Phase Coherent) -> Constructive Interference -> High Norm
    """

    OP_NAME = "complex_dcmt_embedding"

    def __init__(self, cfg, sensor):
        self.fs = sensor.fs
        self.chunk_size = sensor.chunk_size
        
        # Physics config
        self.window_sec = cfg.get("window_sec", 2.0)
        self.hop_chunks = cfg.get("hop_chunks", 4)
        self.bg_alpha   = cfg.get("bg_alpha", 0.01)

        # Buffer setup
        self.n_chunks = int((self.window_sec * self.fs) / self.chunk_size)
        self.buffer = deque(maxlen=self.n_chunks)
        self._hop = 0

        # Model setup
        self.device = cfg.get("device", "cpu")
        emb_dim = cfg.get("embedding_dim", 128)
        
        # NOTE: DcmtModel must accept 'in_channels'. 
        # If your current model file doesn't, you need to update it 
        # to pass in_channels to the first Conv2d layer.
        self.model = DcmtModel(
            emb_dim=emb_dim, 
            in_channels=2  # Real + Imag
        ).to(self.device).eval()

        # State
        self.bg_centroid = None
        self.last_deviation = 0.0
        self.last_norm = 0.0

    @classmethod
    def from_config(cls, cfg, sensor):
        return cls(cfg, sensor)

    def process(self, chunk: np.ndarray, analysis) -> None:
        self.buffer.append(chunk)
        self._hop += 1
        
        # Passthrough metadata for continuity
        analysis.metadata["dcmt_deviation"] = self.last_deviation
        analysis.metadata["dcmt_embedding_norm"] = self.last_norm

        # Wait for buffer fill and hop
        if len(self.buffer) < self.n_chunks:
            return
        if self._hop < self.hop_chunks:
            return
        self._hop = 0

        # --- THE NEW PHYSICS ---
        
        # 1. Get raw complex data (2, F, T)
        raw_complex = analysis.complex_tensor

        # 2. Phase-Preserving Compression
        # Math: x = sign(z) * log(1 + |z|)
        # This reduces dynamic range (like LogMel) but PRESERVES the vector direction (Phase)
        # We process Real/Imag channels independently but using the combined magnitude scaling
        mag = np.sqrt(raw_complex[0]**2 + raw_complex[1]**2) + 1e-9
        scale = np.log1p(mag) / mag
        
        # Apply scale to both channels
        x_in = raw_complex * scale

        # 3. Inference
        x_tensor = torch.from_numpy(x_in).float()
        x_tensor = x_tensor.unsqueeze(0).to(self.device) # (1, 2, F, T)

        with torch.no_grad():
            emb = self.model(x_tensor).cpu().numpy()[0]

        # --- CENTROID LOGIC (Identical to Standard DCMT) ---
        if self.bg_centroid is None:
            self.bg_centroid = emb.copy()
        else:
            self.bg_centroid += self.bg_alpha * (emb - self.bg_centroid)

        deviation = float(np.linalg.norm(emb - self.bg_centroid))
        self.last_deviation = deviation
        self.last_norm = float(np.linalg.norm(emb))

        # Write to metadata (DcmtIsolator will pick this up automatically)
        analysis.metadata["dcmt_deviation"] = deviation
        analysis.metadata["dcmt_embedding_norm"] = self.last_norm
        analysis._dcmt_embedding = emb