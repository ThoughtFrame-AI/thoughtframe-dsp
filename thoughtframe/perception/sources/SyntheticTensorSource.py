# thoughtframe/perception/sources/SyntheticTensorSource.py

import asyncio
import numpy as np
from thoughtframe.perception.interface import TensorSource


class SyntheticTensorSource(TensorSource):
    """
    Deterministic synthetic source for testing perception pipelines.
    Emits items that already contain tensors.
    """

    def __init__(
        self,
        source_id: str,
        shape,
        dtype="float32",
        count: int = 1,
        delay: float = 0.0,
        seed: int | None = None,
    ):
        super().__init__(source_id)
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)
        self.count = count
        self.delay = delay
        self.seed = seed

    @classmethod
    def from_config(cls, cfg):
        return cls(
            source_id=cfg["id"],
            shape=cfg["shape"],
            dtype=cfg.get("dtype", "float32"),
            count=cfg.get("count", 1),
            delay=cfg.get("delay", 0.0),
            seed=cfg.get("seed"),
        )

    async def stream(self):
        rng = np.random.default_rng(self.seed)

        for i in range(self.count):
            tensor = rng.standard_normal(self.shape).astype(self.dtype)

            yield {
                "tensor": tensor,
                "synthetic": True,
                "index": i,
            }

            if self.delay > 0:
                await asyncio.sleep(self.delay)
