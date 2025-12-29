# thoughtframe/sensor/processors/ring_buffer.py

from collections import deque
from queue import Queue
import threading
import time
import os
from numpy import record

import numpy as np
import soundfile as sf
from tf_core.bootstrap import thoughtframe
from thoughtframe.sensor.interface import AcousticChunkProcessor, AcousticAnalysis
from thoughtframe.sensor.mesh_config import THOUGHTFRAME_CONFIG


class RingBufferProcessor(AcousticChunkProcessor):
    
    OP_NAME = "ring_buffer" 
    
    """
    Rolling in-memory audio buffer with asynchronous snapshot persistence.
    """

    def __init__(self, seconds: int, fs: int, chunk_size: int):
        self.seconds = seconds
        self.fs = fs
        self.chunk_size = chunk_size

        self.max_chunks = int((fs * seconds) / chunk_size)
        self._buffer = deque(maxlen=self.max_chunks)

        # async persistence machinery
        self._save_queue = Queue()
        self._worker = threading.Thread(
            target=self._save_worker,
            daemon=True
        )
        self._worker.start()

    @classmethod
    def from_config(cls, cfg, sensor):
        return cls(
            seconds=cfg["seconds"],
            fs=sensor.fs,
            chunk_size=sensor.chunk_size
        )

    def process(self, chunk: np.ndarray, analysis: AcousticAnalysis) -> None:
        # --- fast path ---
        self._buffer.append(chunk)
        analysis.metadata["ring_buffer_chunks"] = len(self._buffer)

        if "saverequested" not in analysis.flags:
            return

        # --- freeze snapshot immediately ---
        snapshot = np.concatenate(list(self._buffer))
        ts = analysis.timestamp or time.time()
        sensor_id = analysis.sensor_id or "unknown"

        self._save_queue.put((snapshot, ts, sensor_id))

        analysis.events.append({
            "type": "audio.snapshot.queued",
            "seconds": self.seconds,
            "chunks": len(self._buffer),
            "timestamp": ts,
        })

    def _save_worker(self):
        """
        Background persistence loop.
        This NEVER runs on the audio path.
        """
        while True:
            snapshot, ts, sensor_id = self._save_queue.get()
            ts = int(ts)

            try:
                ##Maybe save this later?
                ##filename = f"/tmp/audio_snapshot_{sensor_id}_{int(ts)}.npy"
                ##np.save(filename, snapshot)
                saveroot = thoughtframe.resolve_rooted_path(
                    THOUGHTFRAME_CONFIG,
                    THOUGHTFRAME_CONFIG.get("samples", "audio"),
                    sensor_id
                )
                wave = os.path.join(
                    saveroot,
                    f"audio_snapshot_{sensor_id}_{ts}.wav"
                )
                sf.write(wave,snapshot,samplerate=self.fs,subtype="FLOAT")

            finally:
                self._save_queue.task_done()
