import csv, os, time
from thoughtframe.sensor.interface import AcousticChunkProcessor
from thoughtframe.bootstrap import thoughtframe
from thoughtframe.sensor.mesh_config import THOUGHTFRAME_CONFIG
from pytimeparse.timeparse import timeparse
import numpy as np

class SlopeWindowIsolator(AcousticChunkProcessor):
    
    def __init__(self, cfg, sensor):
        super().__init__(cfg, sensor)
        self.state = "BASELINE"
        self.window_id = 0
        self.enter_slope = cfg.get("enter_slope", 0.0005)
        self.exit_slope = cfg.get("exit_slope", 0.0000)
        self.min_duration_sec = timeparse(cfg.get("min_duration", "2s"))
        self.prev_anomaly_rate = 0
        self.sensor = sensor
        
        path = thoughtframe.resolve_rooted_path(
            THOUGHTFRAME_CONFIG,
            THOUGHTFRAME_CONFIG.get("samples", "audio"),
            self.sensor.sensor_id
        )
        
        os.makedirs(path, exist_ok=True)
        
        self.windows_path = os.path.join(path, "slopewindows.csv")
        self.windows_file = open(self.windows_path, "a", newline="")
        self.windows_writer = None

            
    @classmethod
    def from_config(cls, cfg, sensor):
        return cls(cfg,sensor)
  
    def process(self, chunk, analysis):
     
        m = analysis.metadata
        t_sec = analysis.node.t_sec
        anomaly = m.get("anomaly_rate", 0.0)
        delta = anomaly - self.prev_anomaly_rate
        slope = delta / (self.sensor.chunk_size / self.sensor.fs)

        self.prev_anomaly_rate = anomaly       
        print(
           f"t={t_sec:7.1f}  "
           f"anom={anomaly:6.4f}  "
           f"slope={slope: .6f}  "
           f"state={self.state}"
           )
        if self.state == "BASELINE":
            if slope > self.enter_slope :
                print("Slope Event Started")
                self.state = "EVENT"
                self.window_id += 1
                self.window_start_t = t_sec
                self.window_stats = self._init_window_stats()

        elif self.state == "EVENT":     
            duration = t_sec - self.window_start_t
            if slope < self.exit_slope and duration >= self.min_duration_sec:
                print("Slope Event Ended")
                self._flush_window(t_sec)
                self.state = "BASELINE"
                self.window_stats = None
                return
            self._update_window_stats(self.window_stats, m, t_sec )
    
    
    def _flush_window(self, t_sec):
        stats = self.window_stats
        stats["end_t"] = t_sec
        stats["duration"] = t_sec - self.window_start_t
        stats["rms_var"] /= max(1, stats["num_chunks"] - 1)
        stats["centroid_var"] /= max(1, stats["num_chunks"] - 1)
        self._write_window(self.window_stats, self.sensor)  
    
    
    def _init_window_stats(self):
        window_stats = {
            "window_id": self.window_id,
            "start_t": self.window_start_t,
            "end_t": None,
            "duration_sec": None,
            "num_chunks": 0,
        
            "rms_mean": 0.0,
            "rms_var": 0.0,
            "rms_min": np.inf,
            "rms_max": -np.inf,
        
            "centroid_mean": 0.0,
            "centroid_var": 0.0,
        
            "iforest_mean": 0.0,
            "iforest_min": np.inf,
            "fraction_anomalous": 0.0,
        }
        return window_stats
    
    def _update_window_stats(self, stats, m, t_sec):
        stats["num_chunks"] += 1
        n = stats["num_chunks"]
    
        rms = m.get("rms")
        centroid = m.get("spec_centroid_hz")
        iforest = m.get("iforest_score", 0.0)
    
        # RMS
        delta = rms - stats["rms_mean"]
        stats["rms_mean"] += delta / n
        stats["rms_var"] += delta * (rms - stats["rms_mean"])
        stats["rms_min"] = min(stats["rms_min"], rms)
        stats["rms_max"] = max(stats["rms_max"], rms)
    
        # Centroid
        delta_c = centroid - stats["centroid_mean"]
        stats["centroid_mean"] += delta_c / n
        stats["centroid_var"] += delta_c * (centroid - stats["centroid_mean"])
    
        # Isolation forest
        stats["iforest_mean"] += (iforest - stats["iforest_mean"]) / n
        stats["iforest_min"] = min(stats["iforest_min"], iforest)
    

    def _write_window(self, stats):
        if self.windows_writer is None:
            fieldnames = list(stats.keys())
            self.windows_writer = csv.DictWriter(
                self.windows_file,
                fieldnames=fieldnames
            )
            if self.windows_file.tell() == 0:
                self.windows_writer.writeheader()
    
        self.windows_writer.writerow(stats)
        self.windows_file.flush()
       
    
                