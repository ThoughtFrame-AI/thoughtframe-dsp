from abc import ABC, abstractmethod
from typing import AsyncIterator
import numpy as np

class AcousticAnalysis:
    def __init__(self, chunk, node, timestamp=None):
        self.chunk = chunk
        self.node = node
        self.sensor_id = node.sensor.sensor_id
        self.timestamp = timestamp
        self.events: list[dict] = []
        self.metadata: dict = {}
        self.flags: set[str] = set()   
        
    

class AcousticChunkProcessor(ABC):
   
    @abstractmethod
    def process(self, chunk: np.ndarray, analysis: AcousticAnalysis) -> None:
        pass
    @classmethod
    @abstractmethod
    def from_config(cls, cfg, sensor):
        raise NotImplementedError(
            f"{cls.__name__}.from_config() must be implemented"
        )

class AcousticSensor(ABC):
    def __init__(self, sensor_id: str,fs: int, chunk_size: int):
        self.sensor_id = sensor_id
        self.fs = fs
        self.chunk_size = chunk_size

    @abstractmethod
    async def stream(self) -> AsyncIterator[np.ndarray]:
        """
        Async generator yielding np.float32 arrays of shape (chunk_size,)
        """
        pass
    
    @classmethod
    @abstractmethod
    def from_config(cls, cfg):
        raise NotImplementedError(
            f"{cls.__name__}.from_config() must be implemented"
        )

class AcousticPipeline:
    
    def __init__(self):
        self.filters: list[AcousticChunkProcessor] = []
    
    def addChunkProcessor(self, chunkprocessor: AcousticChunkProcessor):
        self.filters.append(chunkprocessor)

    def execute(self, chunk,node):
        analysis :AcousticAnalysis = AcousticAnalysis(chunk,node)
        #print(analysis.flags)
        for processor in self.filters:
            processor.process(chunk,analysis)
        return analysis
