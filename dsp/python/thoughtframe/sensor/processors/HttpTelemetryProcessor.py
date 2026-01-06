import threading
import requests
import time
import logging
from queue import Queue, Empty
import numpy as np
from thoughtframe.sensor.interface import AcousticChunkProcessor, AcousticAnalysis

log = logging.getLogger(__name__)

class HttpTelemetryProcessor(AcousticChunkProcessor):
    OP_NAME = "telemetry_http"

    def __init__(self, cfg, sensor):
        self.sensor_id = getattr(sensor, "sensor_id", "unknown")
        
        # Matches DspModule.java handleTelemetry -> DspManager.handleSensorTelemetry
        self.url = cfg.get("url", "http://localhost:8080/thoughtframe/frames/dsp/api/handletelemetry")
        
        # Batching settings (Java performs a bulk save, so bigger batches = less DB locking)
        self.batch_size = int(cfg.get("batch_size", 50)) 
        self.flush_interval = float(cfg.get("flush_interval", 1.0))
        
        self.queue = Queue(maxsize=1000)
        self.running = True
        
        # Daemon thread ensures this dies when the main app exits
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

    @classmethod
    def from_config(cls, cfg, sensor):
        return cls(cfg, sensor)

    def process(self, chunk: np.ndarray, analysis: AcousticAnalysis) -> None:
        """
        Lightweight extraction. Puts data into queue and returns immediately.
        """
        meta = analysis.metadata
        
        # Keys matched to DspManager.java lines 97-100
        payload = {
            "t": float(analysis.node.t_sec),
            "r": float(meta.get("rms", 0.0)),
            "c": float(meta.get("spec_centroid_hz", 0.0)),
            "s": float(meta.get("iforest_score", 0.0)),
            
            # Java parses this on line 86: source.lastIndexOf("_")
            # Result: beam_id={sensor_id}, isolator="canonical"
            "source": f"{self.sensor_id}_canonical" 
        }

        if not self.queue.full():
            self.queue.put(payload)

    def _worker_loop(self):
        batch = []
        last_flush = time.time()

        while self.running:
            try:
                # Wait for data, but timeout frequently to check flush time
                item = self.queue.get(timeout=0.1)
                batch.append(item)
            except Empty:
                pass

            is_full = len(batch) >= self.batch_size
            is_time = (time.time() - last_flush) > self.flush_interval

            if (is_full or is_time) and batch:
                self._send_batch(batch)
                batch = []
                last_flush = time.time()

    def _send_batch(self, batch):
        # Payload structure matches DspManager.java line 72: inPayload.get("data")
        wrapper = { "data": batch }
        
        try:
            headers = {'Content-Type': 'application/json'}
            resp = requests.post(self.url, json=wrapper, headers=headers, timeout=2)
            if resp.status_code != 200:
                log.warning(f"Telemetry HTTP {resp.status_code}: {resp.text}")
        except Exception as e:
            log.error(f"Telemetry Connection Error: {e}")

    def close(self):
        self.running = False
        self.worker_thread.join()