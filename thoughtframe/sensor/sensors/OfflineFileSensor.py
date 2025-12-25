# thoughtframe/sensor/ffmpeg.py

import asyncio
import subprocess

import numpy as np
from thoughtframe.sensor.interface import AcousticSensor
###https://registry.opendata.aws/pacific-sound/
###https://docs.mbari.org/pacific-sound/notebooks/data/PacificSound2kHz/#retrieve-metadata-for-a-file



class OfflineFileSensor(AcousticSensor):

    def __init__(self, cfg):
        super().__init__(
            sensor_id=cfg["id"],
            fs=cfg.get("fs", 8000),
            chunk_size=cfg.get("chunk_size", 1024),
        )
        self.ffmpeg_cmd = cfg["cmd"]
        self.proc = None
        self.bytes_per_chunk = self.chunk_size * 4  # float32
    
    @classmethod
    
    def from_config(cls, cfg):
        return cls(cfg)

    async def stream(self):
        self.proc = subprocess.Popen(
            self.ffmpeg_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )

        buffer = b""
    
        try:
            while True:
                # BLOCKING read, big chunk
                data = self.proc.stdout.read(self.bytes_per_chunk * 32)
    
                if not data:
                    break
    
                buffer += data
    
                while len(buffer) >= self.bytes_per_chunk:
                    chunk_bytes = buffer[:self.bytes_per_chunk]
                    buffer = buffer[self.bytes_per_chunk:]
    
                    yield np.frombuffer(chunk_bytes, dtype=np.float32)
    
        finally:
            if self.proc:
                self.proc.terminate()
                self.proc.wait()
