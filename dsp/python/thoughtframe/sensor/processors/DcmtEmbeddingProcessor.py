from collections import deque

import torch

import numpy as np
from thoughtframe.models.dcmt_model import DcmtModel
from thoughtframe.sensor.interface import AcousticChunkProcessor


class DcmtEmbeddingProcessor(AcousticChunkProcessor):

    OP_NAME = "dcmt_embedding"

    def __init__(self, cfg, sensor):
        self.fs = sensor.fs
        self.chunk_size = sensor.chunk_size
        self.last_deviation = 0.0
        self.last_norm = 0.0
        self.window_sec = cfg.get("window_sec", 2.0)
        self.hop_chunks = cfg.get("hop_chunks", 4)

        self.n_chunks = int(
            (self.window_sec * self.fs) / self.chunk_size
        )

        self.buffer = deque(maxlen=self.n_chunks)
        self._hop = 0

        self.device = cfg.get("device", "cpu")

        self.model = DcmtModel(
            emb_dim=cfg.get("embedding_dim", 128)
        ).to(self.device).eval()

        self.bg_alpha = cfg.get("bg_alpha", 0.01)
        self.bg_centroid = None

    @classmethod
    def from_config(cls, cfg, sensor):
        return cls(cfg, sensor)

    def process(self, chunk: np.ndarray, analysis) -> None:
        self.buffer.append(chunk)
        self._hop += 1
        analysis.metadata["dcmt_deviation"] = self.last_deviation
        analysis.metadata["dcmt_embedding_norm"] = self.last_norm


        if len(self.buffer) < self.n_chunks:
            return

        if self._hop < self.hop_chunks:
            return

        self._hop = 0

        # --- canonical perception (lazy, shared) ---
        log_mel = analysis.log_mel   # (F, T)

        x = torch.from_numpy(log_mel).float()
        x = x.unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            emb = self.model(x).cpu().numpy()[0]

        # --- background centroid ---
        if self.bg_centroid is None:
            self.bg_centroid = emb.copy()
        else:
            self.bg_centroid += self.bg_alpha * (emb - self.bg_centroid)

        deviation = float(np.linalg.norm(emb - self.bg_centroid))
        self.last_deviation = deviation
        self.last_norm = float(np.linalg.norm(emb))


        # --- small outputs only ---
        analysis.metadata["dcmt_deviation"] = deviation
        analysis.metadata["dcmt_embedding_norm"] = float(
            np.linalg.norm(emb)
        )

        # (optional) keep embedding in-memory only
        analysis._dcmt_embedding = emb
