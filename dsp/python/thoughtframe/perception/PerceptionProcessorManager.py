import inspect

from thoughtframe.perception import processors
from thoughtframe.perception.interface import PerceptionProcessor


class PerceptionProcessorManager:
    def __init__(self):
        self._factories = {}
        self._auto_register()

    def register(self, name, factory):
        self._factories[name] = factory

    def createProcessor(self, cfg, source):
        op = cfg["op"]
        if op not in self._factories:
            raise KeyError(f"Unknown processor '{op}'")
        return self._factories[op].from_config(cfg, source)

    def _auto_register(self):
        for _, obj in inspect.getmembers(processors, inspect.isclass):
            if issubclass(obj, PerceptionProcessor):
                if obj is not PerceptionProcessor and obj.OP_NAME:
                    self._factories[obj.OP_NAME] = obj
