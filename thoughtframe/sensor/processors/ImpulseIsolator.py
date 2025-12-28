# thoughtframe/ml/acoustic_worker.py

from collections import deque
import os
import numpy as np
from thoughtframe.sensor.interface import AcousticAnalysis
from thoughtframe.sensor.processors import WindowIsolator


class ImpulseIsolator(WindowIsolator):
    """
    Window isolator based on impulse density / ICI statistics.
    Includes RAW LOGGING for forensics.
    """

    def __init__(self, cfg, sensor):
        super().__init__(cfg, sensor)
        self.fs = sensor.fs
        self.window_seconds = cfg.get("window_sec", 5.0)

        # rolling impulse timestamps
        self.impulsebuffer = deque()

        # debounce
        self.last_impulse_time = -1.0
        self.refractory_sec = cfg.get("refractory_sec", 0.005)

        # simple threshold for declaring EVENT
        self.min_impulses = cfg.get("min_impulses", 5)
        
        # --- QUICK & DIRTY LOGGING ---
        # Ensure directory exists first
        os.makedirs("thoughtframe/telemetry", exist_ok=True)
        self.raw_log = open("thoughtframe/telemetry/raw_impulses.txt", "w")

    @classmethod
    def from_config(cls, cfg, sensor):
        return cls(cfg, sensor)

    # ------------------------------------------------------------------
    # DSP
    # ------------------------------------------------------------------

    def extract_impulses(self, chunk: np.ndarray):
        x = chunk
        if len(x) < 3:
            return []

        # TKEO
        tkeo = x[1:-1]**2 - x[:-2] * x[2:]
        tkeo = np.abs(tkeo)

        noise_floor = np.median(tkeo)
        if noise_floor <= 0:
            return []

        threshold = noise_floor * self.cfg.get("threshold_mult", 10.0)
        offsets = np.where(tkeo > threshold)[0]

        # center index alignment
        return offsets + 1

    # ------------------------------------------------------------------
    # WINDOW STATE LOGIC
    # ------------------------------------------------------------------

    def propose_state(self, chunk, analysis) -> str:
        """
        Decide whether we are in BASELINE or EVENT based on
        impulse density inside the rolling window.
        """
        t_sec = analysis.node.t_sec
        offsets = self.extract_impulses(chunk)

        for offset in offsets:
            impulse_time = t_sec + offset / self.fs
            if impulse_time - self.last_impulse_time >= self.refractory_sec:
                self.impulsebuffer.append(impulse_time)
                self.last_impulse_time = impulse_time
                
                # --- LOG IT RAW ---
                self.raw_log.write(f"{impulse_time:.6f}\n")
        
        # Flush occasionally to make sure data hits disk if you stop early
        self.raw_log.flush()

        cutoff = t_sec - self.window_seconds
        while self.impulsebuffer and self.impulsebuffer[0] < cutoff:
            self.impulsebuffer.popleft()

        # simple regime decision
        if len(self.impulsebuffer) >= self.min_impulses:
            return "EVENT"
        else:
            return "BASELINE"

    # ------------------------------------------------------------------
    # WINDOW STATS
    # ------------------------------------------------------------------

    def _init_window_stats(self) -> dict:
        return {          
            "num_impulses": 0,
            "ici_mean": None,
            "ici_var": None,
        }

    def _update_window_stats(self, stats: dict, chunk, analysis) -> None:
        n = len(self.impulsebuffer)
        stats["num_impulses"] = n

        if n >= 2:
            icis = np.diff(self.impulsebuffer)
            stats["ici_mean"] = float(np.mean(icis))
            stats["ici_var"] = float(np.var(icis))