# thoughtframe/sensor/processors/time_window_isolator.py

from pytimeparse.timeparse import timeparse
import numpy as np
from thoughtframe.sensor.interface import AcousticChunkProcessor
from thoughtframe.sensor.mesh_config import THOUGHTFRAME_CONFIG
from thoughtframe.sensor.processors.WindowIsolator import WindowIsolator

class TimeWindowIsolator(WindowIsolator): 
    OP_NAME = "time_window_isolator" 

    def __init__(self, cfg, sensor):
        super().__init__(cfg, sensor)
        # Default to 5 minutes if not specified
        self.window_sec = timeparse(cfg.get("width", "5m"))

    def propose_state(self, chunk, analysis) -> str:
        # If the base class hasn't set a start time yet, just continue
        if self.window_start_t is None:
            return "WINDOW"

        t_sec = analysis.node.t_sec
        
        # Check if we have exceeded the time window width
        if (t_sec - self.window_start_t) >= self.window_sec:
            
            # --- CRITICAL FIX: Emit the finalized event ---
            # This tells RingBufferProcessor to flush the audio to disk
            analysis.events.append({
                "type": "time_window.finalized",
                "start_t": self.window_start_t,
                "end_t": t_sec,
            })
            
            return "WINDOW_FLIP"
            
        return "WINDOW"

    def _init_window_stats(self) -> dict:
        return {}

    def _update_window_stats(self, stats, chunk, analysis) -> None:
        pass
    
    @classmethod
    def from_config(cls, cfg, sensor):
        return cls(cfg, sensor)