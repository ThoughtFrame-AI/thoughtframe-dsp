import csv, os, time

from pytimeparse.timeparse import timeparse
from thoughtframe.bootstrap import thoughtframe

import numpy as np
from thoughtframe.sensor.interface import AcousticChunkProcessor
from thoughtframe.sensor.mesh_config import THOUGHTFRAME_CONFIG
from thoughtframe.sensor.processors.WindowIsolator import WindowIsolator


class TimeWindowIsolator(WindowIsolator): 

    def __init__(self, cfg, sensor):
        super().__init__(cfg, sensor)
        self.window_sec = timeparse(cfg.get("width", "5m"))

    def propose_state(self, chunk, analysis) -> str:
        if self.window_start_t is None:
            return "WINDOW"

        t_sec = analysis.node.t_sec
        if (t_sec - self.window_start_t) >= self.window_sec:
            return "WINDOW_FLIP"
        return "WINDOW"

    def _init_window_stats(self) -> dict:
        return {}

    def _update_window_stats(self, stats, chunk, analysis) -> None:
        pass
    
    @classmethod
    def from_config(cls, cfg, sensor):
        return cls(cfg, sensor)