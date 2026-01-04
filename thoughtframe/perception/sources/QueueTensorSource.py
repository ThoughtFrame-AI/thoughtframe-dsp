# thoughtframe/perception/sources/QueueTensorSource.py

import asyncio
from typing import Any
from thoughtframe.perception.interface import TensorSource


class QueueTensorSource(TensorSource):
    def __init__(self, source_id: str, maxsize: int = 0):
        super().__init__(source_id)
        self._queue = asyncio.Queue(maxsize=maxsize)
        self._closed = False

    @classmethod
    def from_config(cls, cfg):
        return cls(
            source_id=cfg["id"],
            maxsize=cfg.get("maxsize", 0),
        )

    def push(self, item: Any):
        if self._closed:
            raise RuntimeError("QueueTensorSource is closed")
        self._queue.put_nowait(item)

    def close(self):
        self._closed = True
        self._queue.put_nowait(None)  # sentinel

    async def stream(self):
        while True:
            item = await self._queue.get()
            if item is None:
                break
            yield item
