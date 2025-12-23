from thoughtframe.sensor.processors import *


class SensorProcessorManager:
    def __init__(self):
        self._factories = {}
        self.register_processors()

    def register(self, name, factory):
        self._factories[name] = factory

    def createProcessor(self, cfg, sensor):
        op = cfg["op"]
        if op not in self._factories:
            raise KeyError(f"Unknown processor '{op}'")
        return self._factories[op].from_config(cfg, sensor)


    def register_processors(self):
        self.register("debug", DebugProbeProcessor)
        self.register("ring_buffer", RingBufferProcessor)
        self.register("isolation_forest", IsolationForestProcessor)
        self.register("snapshot", SnapshotProcessor)