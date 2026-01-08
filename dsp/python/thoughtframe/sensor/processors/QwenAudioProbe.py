import torch
from transformers.models.auto.processing_auto import AutoProcessor

import numpy as np
from thoughtframe.sensor.interface import AcousticChunkProcessor
from transformers.models.qwen2_audio.modeling_qwen2_audio import (
    Qwen2AudioForConditionalGeneration
)


class QwenAudioProbe(AcousticChunkProcessor):
    OP_NAME = "qwen_audio"

    def __init__(self, cfg, sensor):

        self.sensor = sensor
        self.buffer = []
        self.max_seconds = cfg.get("seconds", 30)
        self.max_samples = int(self.max_seconds * sensor.fs)

        self.model = None
        self.processor = None
        self.initialized = False

    @classmethod
    def from_config(cls, cfg, sensor):
        return cls(cfg, sensor)

    def process(self, chunk, analysis):
        # accumulate raw audio
        self.buffer.append(chunk)

        total_samples = sum(len(c) for c in self.buffer)
        if total_samples < self.max_samples:
            return

        audio = np.concatenate(self.buffer)
        self.buffer.clear()

        semantic = self.run_qwen(audio)

        analysis.metadata.setdefault("semantic", {})
        analysis.metadata["semantic"]["qwen"] = semantic
        analysis.metadata["semantic"]["qwen"]["chunk_index"] = analysis.chunk_index


    def run_qwen(self, audio: np.ndarray):

        if not self.initialized:
            model_id = "Qwen/Qwen2-Audio-7B"
    
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if self.device == "cuda" else torch.float32
    
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
                model_id,
                dtype=dtype,
                device_map="auto" if self.device == "cuda" else None,
            )
            self.model.eval()
            self.initialized = True
    
        # Keep your known-good prompt + audios= path
        prompt = (
            "Listen carefully to this audio segment. "
            "Describe what is happening over time. "
            "Focus on changes, patterns, and distinct sound sources. "
            "Do not assume this is speech or music."
        )
    
        inputs = self.processor(
            text=prompt,
            audios=audio,
            sampling_rate=self.sensor.fs,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
    
        with torch.no_grad():
            # Forward pass only: no generation, request hidden states
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )
    
        # -----------------------------
        # Robust extraction:
        # Try audio-specific first, then encoder, then generic hidden_states
        # -----------------------------
    
        hs = None
        source = None
    
        if hasattr(outputs, "audio_hidden_states") and outputs.audio_hidden_states is not None:
            hs = outputs.audio_hidden_states
            source = "audio_hidden_states"
        elif hasattr(outputs, "encoder_hidden_states") and outputs.encoder_hidden_states is not None:
            hs = outputs.encoder_hidden_states
            source = "encoder_hidden_states"
        elif hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            hs = outputs.hidden_states
            source = "hidden_states"
    
        if hs is None:
            # Print available fields once to see what this build returns
            keys = list(outputs.keys()) if hasattr(outputs, "keys") else dir(outputs)
            raise RuntimeError(f"No hidden states found. Output fields: {keys}")
    
        # hs is typically a tuple: [layer0, layer1, ...] each [B, T, D] (or similar)
        layer_index = len(hs) // 2
        H = hs[layer_index][0]  # [T, D] for batch 0
    
        # Mean-pool over time → single vector
        emb_t = H.mean(dim=0).float().cpu().numpy()
    
        # Quick sanity
        import numpy as np
        print(f"[QWEN-VEC] src={source} layer={layer_index} shape={emb_t.shape} norm={np.linalg.norm(emb_t):.4f}")
    
        return {
            "embedding": emb_t,
            "source": source,
            "layer": layer_index,
            "frames": int(H.shape[0]),
            "dim": int(emb_t.shape[0]),
            "duration_sec": len(audio) / self.sensor.fs,
        }
