from collections import deque
import requests
import os
import csv
import threading
from queue import Queue
import numpy as np
import soundfile as sf
from pytimeparse.timeparse import timeparse

from tf_core.bootstrap import thoughtframe
from thoughtframe.sensor.interface import AcousticChunkProcessor, AcousticAnalysis
from thoughtframe.sensor.mesh_config import THOUGHTFRAME_CONFIG

class RingBufferProcessor(AcousticChunkProcessor):
    OP_NAME = "ring_buffer"
    
    # 1. FIX: Define strict column order to prevent "gibberish" CSVs
    CSV_FIELDS = [
        "window_id", "start_t", "end_t", "start_chunk", "end_chunk",
        "duration_sec", "duration_chunks", "num_chunks",
        "rms_mean", "rms_var", "rms_min", "rms_max",
        "centroid_mean", "centroid_var"
    ]

    def __init__(self,cfg, window_sec: int, fs: int, chunk_size: int, sensor_id: str):
        self.fs = fs
        self.chunk_size = chunk_size
        self.sensor_id = sensor_id
        self.window_limit = window_sec

        self._java_endpoint = cfg.get( "guard_endpoint", "http://localhost:8080/thoughtframe/lab/dsp/api/handlewindowevent" )

        # Buffer sizing (Window + 50% headroom for safety)
        buffer_len_sec = window_sec * 1.5
        self.max_chunks = int((fs * buffer_len_sec) / chunk_size)
        self._buffer = deque(maxlen=self.max_chunks)

        # State tracking
        self.window_id = 0
        self.window_start_t = None
        self.window_start_chunk = None
        self._reset_stats()

        # Paths
        self.output_root = thoughtframe.resolve_rooted_path(
            THOUGHTFRAME_CONFIG,
            THOUGHTFRAME_CONFIG.get("samples", "audio"),
            self.sensor_id
        )
        os.makedirs(self.output_root, exist_ok=True)
        
        self.csv_path = os.path.join(self.output_root, "TimeWindowStats.csv")
        self._init_csv()

        # Async persistence
        self._save_queue = Queue(maxsize=4)        
        self._worker = threading.Thread(target=self._save_worker, daemon=True)
        self._worker.start()

    @classmethod
    def from_config(cls, cfg, sensor):
        width_str = cfg.get("window_width") or cfg.get("width", "5m")
        return cls(cfg,
            window_sec=timeparse(width_str),
            fs=sensor.fs,
            chunk_size=sensor.chunk_size,
            sensor_id=sensor.sensor_id
        )

    def process(self, chunk: np.ndarray, analysis: AcousticAnalysis) -> None:
        t_sec = analysis.node.t_sec
        ci = analysis.chunk_index
        
        # Initialize first window if needed
        if self.window_start_t is None:
            self.window_start_t = t_sec
            self.window_start_chunk = ci
            self.window_id = 1

        # A. Buffer Audio (No calculation, just storage)
        self._buffer.append(chunk)

        # B. Aggregate Existing Stats
        # We grab the values calculated by SpectralFeatureProcessor
        self._accumulate_stats(analysis)

        # C. Check Time Limit
        if (t_sec - self.window_start_t) >= self.window_limit:
            self._finalize_window(t_sec, ci)

    def _finalize_window(self, end_t, end_chunk):
        """Snapshots data and resets state for the next continuous window."""
        
        # 1. Snapshot Audio
        audio_snapshot = np.concatenate(self._buffer)
        
        # 2. Snapshot Stats
        stats_snapshot = self.stats.copy()
        stats_snapshot.update({
            "window_id": self.window_id,
            "start_t": self.window_start_t,
            "end_t": end_t,
            "start_chunk": self.window_start_chunk,
            "end_chunk": end_chunk,
            "duration_sec": end_t - self.window_start_t,
            "duration_chunks": (end_chunk - self.window_start_chunk) + 1
        })

        # 3. Queue Saves (Background Thread)
        self._save_queue.put({
            "type": "audio",
            "data": audio_snapshot,
            "filename": f"timewindow_{self.sensor_id}_{self.window_start_chunk}_{end_chunk}.flac"
        })
        
        self._save_queue.put({
            "type": "csv",
            "data": stats_snapshot
        })

        self._save_queue.put({
            "type": "event",
            "data": stats_snapshot
        })

        # 4. Reset for Next Window (Immediate continuity)
        self.window_start_t = end_t
        self.window_start_chunk = end_chunk + 1
        self.window_id += 1
        self._buffer.clear()
        self._reset_stats()
        
    def _emit_window_event(self, stats):
        payload = {
            "event": "WINDOW_FINALIZED",
            "source": self.sensor_id,
            "isolator": "timewindow",
            "window_id": f"timewindow_{self.sensor_id}_{stats['start_chunk']}_{stats['end_chunk']}",
            "data": stats
        }
        try:
            requests.post(self._java_endpoint, json=payload, timeout=0.5)
        except Exception as e:
            # Never block DSP
            pass

    # ------------------------------------------------------------------
    # STATS AGGREGATION (Only reads analysis.metadata)
    # ------------------------------------------------------------------
    def _reset_stats(self):
        self.stats = {
            "num_chunks": 0,
            "rms_mean": 0.0, "rms_var": 0.0, "rms_min": np.inf, "rms_max": -np.inf,
            "centroid_mean": 0.0, "centroid_var": 0.0
        }

    def _accumulate_stats(self, analysis):
        """
        Reads pre-calculated features from metadata and updates 
        running averages (Welford's Algorithm).
        """
        m = analysis.metadata
        s = self.stats
        s["num_chunks"] += 1
        n = s["num_chunks"]

        # RMS (Expected from SpectralFeatureProcessor)
        rms = m.get("rms")
        if rms is not None:
            if n == 1:
                s["rms_mean"] = rms
                s["rms_min"] = rms
                s["rms_max"] = rms
            else:
                delta = rms - s["rms_mean"]
                s["rms_mean"] += delta / n
                s["rms_var"] += delta * (rms - s["rms_mean"])
                s["rms_min"] = min(s["rms_min"], rms)
                s["rms_max"] = max(s["rms_max"], rms)

        # Spectral Centroid (Expected from SpectralFeatureProcessor)
        centroid = m.get("spec_centroid_hz")
        if centroid is not None:
            if n == 1:
                s["centroid_mean"] = centroid
            else:
                delta_c = centroid - s["centroid_mean"]
                s["centroid_mean"] += delta_c / n
                s["centroid_var"] += delta_c * (centroid - s["centroid_mean"])

    # ------------------------------------------------------------------
    # PERSISTENCE WORKER
    # ------------------------------------------------------------------
    def _init_csv(self):
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.CSV_FIELDS)
                writer.writeheader()

    def _save_worker(self):
        while True:
            task = self._save_queue.get()
            try:
                if task["type"] == "audio":
                    path = os.path.join(self.output_root, task["filename"])
                    sf.write(path, task["data"], samplerate=self.fs, format="FLAC", subtype="PCM_24")
                
                elif task["type"] == "csv":
                    with open(self.csv_path, "a", newline="") as f:
                        # FIX: Enforce fieldnames so columns align with header
                        writer = csv.DictWriter(f, fieldnames=self.CSV_FIELDS, extrasaction='ignore')
                        writer.writerow(task["data"])
                
                elif task["type"] == "event":
                    self._emit_window_event(task["data"])

            except Exception as e:
                print(f"Error in RingBuffer worker: {e}")
            finally:
                self._save_queue.task_done()