from thoughtframe.perception.sources import *



class PerceptionManager:
    def __init__(self):
        self._factories = {}
        self.register_sources()

    def register(self, name, factory):
        self._factories[name] = factory
 
    def createSource(self, cfg):
        source_type = cfg["type"]
        if source_type not in self._factories:
            raise KeyError(f"Unknown source type '{source_type}'")
        return self._factories[source_type].from_config(cfg)

    def register_sources(self):
        self.register("queue", QueueTensorSource)
        self.register("synthetic", SyntheticTensorSource)
