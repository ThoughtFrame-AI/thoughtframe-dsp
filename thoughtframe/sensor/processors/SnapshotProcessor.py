import json
import os
from queue import Queue
import threading
import time

from thoughtframe.sensor.interface import AcousticChunkProcessor
from thoughtframe.sensor.mesh_config import MESH_CONFIG
from thoughtframe.sensor.mesh_config import THOUGHTFRAME_CONFIG
from tf_core.bootstrap import thoughtframe


class SnapshotProcessor(AcousticChunkProcessor):
    
    OP_NAME = "snapshot" 
    
    def __init__(self, cfg, sensor):
        self.config = cfg
        self.sensor = sensor
        self._save_queue = Queue()
        self._worker = threading.Thread(
            target=self._save_worker,
            daemon=True
        )
        self._worker.start()
        print("SnapshotProcessor worker started")

    
    def process(self, chunk, analysis):
        analysis.metadata["chunk_len"] = len(chunk)
        if "saverequested" not in analysis.flags:
            return
        self._save_queue.put(analysis)
        analysis.flags.add("analysis-saved")
    
    
    @classmethod    
    def from_config(cls, cfg, sensor):            
        return cls(cfg, sensor)    
    
    def _save_worker(self):
        print(">>> ENTERED _save_worker", self, SnapshotProcessor)

        """
        Background persistence loop.
        This NEVER runs on the audio path.
        """
        while True:
            analysis = self._save_queue.get()
   
            try:
                record = {
                    "timestamp": analysis.timestamp or time.time(),
                    "sensor_id": analysis.sensor_id,
                    "fs": self.sensor.fs,
                    "chunk_size": self.sensor.chunk_size,
                    "metadata": analysis.metadata.copy(),
                    "flags": list(analysis.flags),
                    "events": list(analysis.events),
                }
                ts = int(record["timestamp"])
                saveroot = thoughtframe.resolve_rooted_path(
                    THOUGHTFRAME_CONFIG,
                    THOUGHTFRAME_CONFIG.get("samples", "audio"),
                    record["sensor_id"]
                )
                path = os.path.join(
                    saveroot,
                    f"audio_snapshot_{record['sensor_id']}_{ts}.json"
                )
                os.makedirs(saveroot, exist_ok=True)

                with open(path, "w") as f:
                    json.dump(record, f, indent=2)
                ##turn analysis into 
                print("Saving analysis")
            except Exception as e:
                print("crashed" ,e)
                raise                   
            finally:
                self._save_queue.task_done()