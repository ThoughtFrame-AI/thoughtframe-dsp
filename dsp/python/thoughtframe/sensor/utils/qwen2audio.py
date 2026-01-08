import numpy as np
from thoughtframe.sensor.interface import AcousticChunkProcessor

class QwenAudioProcessor(AcousticChunkProcessor):
    OP_NAME = "qwen_audio"

    def __init__(self, cfg, sensor):
        super().__init__(cfg, sensor)

        self.buffer = []
        self.max_seconds = cfg.get("seconds", 30)
        self.max_samples = int(self.max_seconds * sensor.fs)

        self.model = None
        self.processor = None
        self.initialized = False

    @classmethod
    def from_config(cls, cfg, sensor):
        return cls(cfg, sensor)

    def process_chunk(self, chunk, analysis, node):
        # accumulate raw audio
        self.buffer.append(chunk)

        total_samples = sum(len(c) for c in self.buffer)
        if total_samples < self.max_samples:
            return analysis

        audio = np.concatenate(self.buffer)
        self.buffer.clear()

        semantic = self.run_qwen(audio)

        analysis.metadata.setdefault("semantic", {})
        analysis.metadata["semantic"]["qwen"] = semantic
        analysis.metadata["semantic"]["qwen"]["t_sec"] = node.t_sec

        return analysis

    def run_qwen(self, audio):
        """
        Stub for now – replace with real Qwen call once stable.
        """
        return {
            "status": "buffer_complete",
            "samples": len(audio),
            "duration_sec": len(audio) / self.sensor.fs
        }
