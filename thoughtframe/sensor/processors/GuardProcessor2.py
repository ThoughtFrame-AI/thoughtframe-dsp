import numpy as np
import multiprocessing as mp
from multiprocessing import shared_memory
from types import SimpleNamespace
import queue

from thoughtframe.sensor.interface import (
    AcousticChunkProcessor,
    AcousticAnalysis,
)

# ============================================================
# OPTIMIZED WORKER-SIDE SPECTRAL FILTER (IN-PLACE)
# ============================================================

def _apply_guard_filter(
    x: np.ndarray,
    analysis: AcousticAnalysis,
    *,
    lpf=None,
    hpf=None,
    band=None,
    **kwargs
) -> np.ndarray:
    """
    Frequency-domain filter applied inside worker process.
    """
    if not (lpf or hpf or band):
        return x

    n = x.shape[0]
    X = analysis.fft
    freqs = analysis.fft_freqs

    if band:
        low, high = band
        mask = (freqs >= low) & (freqs <= high)
    elif hpf:
        mask = freqs >= hpf
    elif lpf:
        mask = freqs <= lpf
    else:
        return x

    # Filter in the frequency domain
    X_filtered = X * mask
    
    # Inverse FFT back to time domain
    y = np.fft.irfft(X_filtered, n=n)

    # Cast back to original dtype without extra allocation
    return y.astype(x.dtype, copy=False)

# ============================================================
# WORKER PROCESS ENTRYPOINT (SHARED MEMORY)
# ============================================================

def _guard_worker_entry(
    q: mp.Queue,
    shm_name: str,
    slots: int,
    cfg: dict,
    fs: int,
    chunk_size: int,
    sensor_id: str
):
    # 1. Attach to shared memory block by name
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    bytes_per_chunk = chunk_size * np.dtype(np.float32).itemsize

    # 2. Local Factory & Context Setup
    from thoughtframe.sensor.SensorProcessorManager import SensorProcessorManager
    
    worker_context = SimpleNamespace(
        id=sensor_id,
        sensor_id=sensor_id,
        fs=fs,
        chunk_size=chunk_size,
        create_node=lambda c_idx, t: SimpleNamespace(
            chunk_index=c_idx,
            t_sec=t,
            sensor=worker_context 
        )
    )

    # 3. Instantiate child processors locally
    pm = SensorProcessorManager()
    children = []
    for child_cfg in cfg.get("pipeline", []):
        child_cfg = dict(child_cfg)
        child_cfg["csv_prefix"] = cfg.get("name", "guard")
        child = pm.createProcessor(child_cfg, worker_context)
        children.append(child)

    # 4. Worker loop (Blocks on Q until data is ready in SHM)
    try:
        while True:
            msg = q.get()
            if msg is None: break

            # Map the specific slot in shared memory back to a numpy array
            offset = msg["slot"] * bytes_per_chunk
            chunk = np.ndarray(
                (chunk_size,), 
                dtype=np.float32, 
                buffer=existing_shm.buf, 
                offset=offset
            )

            # Recreate node + analysis locally using our mock
            node = worker_context.create_node(msg["chunk_index"], msg["t_sec"])
            analysis = AcousticAnalysis(chunk, node)

            # Apply Filter (In-place if possible)
            filtered = _apply_guard_filter(chunk, analysis, **cfg)

            # Execute Children
            for child in children:
                child.process(filtered, analysis)
                
    finally:
        existing_shm.close()


# ============================================================
# GUARD PROCESSOR (PRODUCER)
# ============================================================

class GuardProcessor(AcousticChunkProcessor):
    OP_NAME = "guard"

    def __init__(self, cfg: dict, sensor):
        self.cfg = cfg
        self.sensor = sensor
        self.name = cfg.get("name", "guard")

        # 1. PRE-ALLOCATE SHARED MEMORY RING BUFFER
        # Default to 64 slots for smoother at-sea performance
        self.slots = cfg.get("queue_depth", 64)
        self.bytes_per_chunk = sensor.chunk_size * np.dtype(np.float32).itemsize
        self.shm_size = self.bytes_per_chunk * self.slots
        
        # Create a unique block of memory
        self.shm = shared_memory.SharedMemory(create=True, size=self.shm_size)
        
        # Blocking queue only holds indices, not heavy data
        self._q = mp.Queue(maxsize=self.slots)

        worker_args = (
            self._q,
            self.shm.name,
            self.slots,
            cfg,
            sensor.fs,
            sensor.chunk_size,
            getattr(sensor, "sensor_id", "unknown")
        )

        self._worker = mp.Process(
            target=_guard_worker_entry,
            args=worker_args,
            daemon=True,
        )
        self._worker.start()
        self._current_slot = 0

    @classmethod
    def from_config(cls, cfg, sensor):
        return cls(dict(cfg), sensor)

    def process(self, chunk: np.ndarray, analysis: AcousticAnalysis) -> None:
        if chunk.size == 0: return

        # 2. ZERO-COPY WRITE TO SHARED MEMORY
        offset = self._current_slot * self.bytes_per_chunk
        # Point a local numpy view at the shared memory slot
        target = np.ndarray(
            chunk.shape, 
            dtype=chunk.dtype, 
            buffer=self.shm.buf, 
            offset=offset
        )
        # Fast memory copy into the shared slot
        target[:] = chunk[:] 

        # 3. BLOCKING PUT (Zero Loss Backpressure)
        # This blocks the main loop ONLY if the worker is truly maxed out.
        # This is the "As fast as possible but no faster" limit.
        self._q.put({
            "slot": self._current_slot,
            "chunk_index": analysis.node.chunk_index,
            "t_sec": analysis.node.t_sec,
        })

        # Advance to next slot in ring buffer
        self._current_slot = (self._current_slot + 1) % self.slots

    def __del__(self):
        # Graceful cleanup to prevent memory leaks in the OS
        if hasattr(self, '_q'):
            self._q.put(None)
        if hasattr(self, 'shm'):
            self.shm.close()
            self.shm.unlink()