# thoughtframe/sensor/arrays/ArrayCoordinator.py

import asyncio

import numpy as np
from thoughtframe.sensor.interface import AcousticSensor
from thoughtframe.sensor.sensors.QueueSensor import QueueSensor


class ArrayCoordinator:
    """
    Owns consumption of N input sensors.
    Produces M beam sensors (QueueSensor).
    """

    def __init__(self, cfg: dict, inputs: list[AcousticSensor]):
        self.cfg = cfg
        self.inputs = inputs

        if not inputs:
            raise ValueError("ArrayCoordinator requires at least one input sensor")

        # Assume homogeneous inputs (enforced by config)
        fs = inputs[0].fs
        chunk_size = inputs[0].chunk_size

        # Create beam sensors
        self.beam_sensors: dict[str, QueueSensor] = {}

        for beam_cfg in cfg.get("beams", []):
            beam_id = beam_cfg["id"]
            self.beam_sensors[beam_id] = QueueSensor(
                sensor_id=beam_id,
                fs=fs,
                chunk_size=chunk_size
            )

    def get_beam_sensor(self, beam_id: str) -> QueueSensor:
        try:
            return self.beam_sensors[beam_id]
        except KeyError:
            raise KeyError(f"Unknown beam id '{beam_id}'")

    async def run(self):
        iters = [sensor.stream().__aiter__() for sensor in self.inputs]
    
        chunk_count = 0
        print(f"[ArrayCoordinator] START inputs={[s.sensor_id for s in self.inputs]}")
    
        try:
            while True:
                try:
                    chunks = await asyncio.gather(*[anext(it) for it in iters])
                except StopAsyncIteration:
                    print("[ArrayCoordinator] INPUT STREAM ENDED")
                    break
    
                out_chunk = chunks[0]
    
                for beam in self.beam_sensors.values():
                    beam.push(out_chunk)
    
                chunk_count += 1
                if chunk_count % 100 == 0:
                    # print(f"[ArrayCoordinator] pushed {chunk_count} chunks")
                    ...
    
        except asyncio.CancelledError:
            print("[ArrayCoordinator] CANCELLED")
            raise
    
        except Exception as e:
            print("[ArrayCoordinator] ERROR:", repr(e))
            raise
    
        finally:
            print("[ArrayCoordinator] SHUTDOWN — closing beams")
            for beam in self.beam_sensors.values():
                beam.close()
    
