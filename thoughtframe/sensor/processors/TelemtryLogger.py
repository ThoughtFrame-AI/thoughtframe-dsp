import csv, os, time
from thoughtframe.sensor.interface import AcousticChunkProcessor
from thoughtframe.bootstrap import thoughtframe
from thoughtframe.sensor.mesh_config import THOUGHTFRAME_CONFIG

class TelemtryLogger(AcousticChunkProcessor):
    OP_NAME = "telemetry" 

    def __init__(self, cfg, sensor):
        self.sensor = sensor
        self.start_ts = None

        path = thoughtframe.resolve_rooted_path(
            THOUGHTFRAME_CONFIG,
            THOUGHTFRAME_CONFIG.get("samples", "audio"),
            sensor.sensor_id
        )
        os.makedirs(path, exist_ok=True)

        self.csv_path = os.path.join(path, "telemetry.csv")
        self.f = open(self.csv_path, "a", newline="")
        self.w = csv.writer(self.f)

        if self.f.tell() == 0:
            self.w.writerow([
                "t_sec",
                "rms",
                "rms_mean",
                "rms_var",
                "spec_centroid_hz",
                "centroid_mean",
                "iforest_score",
                "anomaly_rate"
            ])
            
    @classmethod
    def from_config(cls, cfg, sensor):
        return cls(cfg,sensor)
  
    def process(self, chunk, analysis):
        m = analysis.metadata
        t_sec = analysis.node.t_sec
        self.w.writerow([
           t_sec,
            m.get("rms"),
            m.get("rms_mean"),
            m.get("rms_var"),
            m.get("spec_centroid_hz"),
            m.get("centroid_mean"),
            m.get("iforest_score"),
            m.get("anomaly_rate")
        ])
        self.f.flush()