from thoughtframe.sensor.sensors import *


class SensorManager:
    def __init__(self):
        self._factories = {}
        self.register_processors()

    def register(self, name, factory):
        self._factories[name] = factory

    def createSensor(self, cfg):
        sensor_type = cfg["type"]
        if sensor_type not in self._factories:
            raise KeyError(f"Unknown sensor type '{sensor_type}'")
        return self._factories[sensor_type].from_config(cfg)


    def register_processors(self):
        self.register("ffmpeg", FfmpegAcousticSensor)
        self.register("synthetic", SyntheticAcousticSensor)
        