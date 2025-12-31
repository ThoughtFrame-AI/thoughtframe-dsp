# thoughtframe/sensor/processors/IsolationForestWindowIsolator.py

import numpy as np
from thoughtframe.sensor.processors.WindowIsolator import WindowIsolator


class IsolationForestWindowIsolator(WindowIsolator):
    
    OP_NAME = "if_window_isolator" 

    """
    Window isolator driven by Isolation Forest anomaly score.
    """

    def __init__(self, cfg, sensor):
        super().__init__(cfg, sensor)
        self.threshold = cfg.get("threshold", -0.1)
        self.state = "BASELINE"
        # hysteresis
        self.enter_threshold = self.threshold * 1.1   # more negative
        self.exit_threshold  = self.threshold * 0.9   # less negative

    @classmethod
    def from_config(cls, cfg, sensor):
        return cls(cfg, sensor)

    # ------------------------------------------------------------------
    # STATE DECISION
    # ------------------------------------------------------------------

    def propose_state(self, chunk, analysis) -> str:
        IF = analysis.metadata.get("iforest_score", 0.0)

        if self.state == "BASELINE":
            return "EVENT" if IF < self.enter_threshold else "BASELINE"
        else:
            return "BASELINE" if IF > self.exit_threshold else "EVENT"

    # ------------------------------------------------------------------
    # WINDOW STATS
    # ------------------------------------------------------------------

    def _init_window_stats(self) -> dict:
        return {
            "iforest_mean": 0.0,
            "iforest_min": np.inf,

        }

    def _update_window_stats(self, stats: dict, chunk, analysis) -> None:
        m = analysis.metadata

        n = stats["num_chunks"]

       
        iforest = m.get("iforest_score", 0.0)

    
        # --- Isolation Forest ---
        stats["iforest_mean"] += (iforest - stats["iforest_mean"]) / n
        stats["iforest_min"] = min(stats["iforest_min"], iforest)
