# thoughtframe/sensor/synthetic.py

import asyncio
import time

import numpy as np
from thoughtframe.sensor.interface import AcousticSensor


class SyntheticAcousticSensor(AcousticSensor):
    def __init__(self,sensor_id, fs=8000, chunk_size=1024):
        super().__init__(sensor_id,fs, chunk_size)
        self.t = 0.0



    async def stream(self):
        dt = self.chunk_size / self.fs

        while True:
            samples = np.random.randn(self.chunk_size).astype(np.float32) * 0.1

            if int(time.time()) % 15 < 3:
                freq = 300
                t_axis = np.arange(self.chunk_size) / self.fs
                samples += 0.5 * np.sin(2 * np.pi * freq * (t_axis + self.t))

            self.t += dt
            yield samples

            await asyncio.sleep(dt)
