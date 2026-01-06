from abc import ABC, abstractmethod
from typing import AsyncIterator, Any
import numpy as np


import numpy as np
from typing import Any

class PerceptionAnalysis:
    def __init__(self, item: Any, node, index=None, timestamp=None):
        self.item = item              # original work envelope
        self.tensor: np.ndarray | None = None

        self.node = node
        self.source_id = node.source.source_id
        self.index = index
        self.timestamp = timestamp

        self.events: list[dict] = []
        self.metadata: dict[str, Any] = {}
        self.flags: set[str] = set()

        self._flat = None
        self._normalized = None

    @property
    def shape(self):
        return self.tensor.shape

    @property
    def dtype(self):
        return self.tensor.dtype

    @property
    def flat(self):
        if self._flat is None:
            self._flat = self.tensor.reshape(-1)
        return self._flat

    @property
    def normalized(self):
        if self._normalized is None:
            t = self.tensor.astype(np.float32)
            mean = t.mean()
            std = t.std() + 1e-8
            self._normalized = (t - mean) / std
        return self._normalized




class PerceptionProcessor(ABC):
    OP_NAME: str | None = None

    @abstractmethod
    def process(self, analysis: PerceptionAnalysis) -> None:
        pass

    @classmethod
    @abstractmethod
    def from_config(cls, cfg, source):
        raise NotImplementedError


class TensorSource(ABC):
    """
    Async tensor producer.
    Mirrors AcousticSensor without time / fs semantics.
    """

    def __init__(self, source_id: str):
        self.source_id = source_id

    @abstractmethod
    async def stream(self) -> AsyncIterator[np.ndarray]:
        """
        Async generator yielding np.ndarray tensors.
        """
        pass

    @classmethod
    @abstractmethod
    def from_config(cls, cfg):
        raise NotImplementedError(
            f"{cls.__name__}.from_config() must be implemented"
        )

# thoughtframe/perception/PerceptionPipeline.py



class PerceptionPipeline:
    def __init__(self):
        self.filters: list[PerceptionProcessor] = []

    def addProcessor(self, processor: PerceptionProcessor):
        self.filters.append(processor)

    def execute(self, item, node):
        analysis = PerceptionAnalysis(item, node)

        for processor in self.filters:
            processor.process(analysis)

        return analysis

