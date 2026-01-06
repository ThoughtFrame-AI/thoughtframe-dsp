# thoughtframe/sensor/QueueSensor.py

import asyncio
import numpy as np
from thoughtframe.sensor.interface import AcousticSensor


class QueueSensor(AcousticSensor):
    def __init__(self, sensor_id: str, fs: int, chunk_size: int, maxsize: int = 0):
        super().__init__(sensor_id, fs, chunk_size)
        self._queue = asyncio.Queue(maxsize=maxsize)
        self._closed = False

    @classmethod
    def from_config(cls, cfg):
        return cls(
            sensor_id=cfg["sensor_id"],
            fs=cfg["fs"],
            chunk_size=cfg["chunk_size"],
            maxsize=cfg.get("maxsize", 0),
        )

    def push(self, chunk: np.ndarray):
        if self._closed:
            raise RuntimeError("QueueSensor is closed")
        self._queue.put_nowait(chunk)

    def close(self):
        self._closed = True
        self._queue.put_nowait(None)  # sentinel

    async def stream(self):
        while True:
            chunk = await self._queue.get()
            if chunk is None:
                break
            yield chunk
