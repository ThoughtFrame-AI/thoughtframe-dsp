import inspect

from thoughtframe.sensor import processors
from thoughtframe.sensor.interface import AcousticChunkProcessor


class SensorProcessorManager:
    def __init__(self):
        self._factories = {}
        ##self.register_processors()
        self._auto_register()

    def register(self, name, factory):
        self._factories[name] = factory

    def createProcessor(self, cfg, sensor):
        op = cfg["op"]
        if op not in self._factories:
            raise KeyError(f"Unknown processor '{op}'")
        return self._factories[op].from_config(cfg, sensor)

    def _auto_register(self):
        for _, obj in inspect.getmembers(processors, inspect.isclass):
            if issubclass(obj, AcousticChunkProcessor):
                if obj is not AcousticChunkProcessor and obj.OP_NAME:
                    self._factories[obj.OP_NAME] = obj


    # def register_processors(self): 
    #     self.register("debug", DebugProbeProcessor)
    #     self.register("ring_buffer", RingBufferProcessor)
    #     self.register("isolation_forest", IsolationForestProcessor)
    #     self.register("snapshot", SnapshotProcessor)
    #     self.register("spectral_features", SpectralFeatureProcessor)
    #     self.register("temporal_context", TemporalContextProcessor)
    #     self.register("telemetry", TelemtryLogger)
    #     self.register("window_isolator", WindowIsolator)
    #     self.register("slope_window_isolator", SlopeWindowIsolator)
    #     self.register("time_window_isolator", TimeWindowIsolator)
    #     self.register("if_window_isolator", IsolationForestWindowIsolator)
    #    self.register("impulse_isolator", ImpulseIsolator)