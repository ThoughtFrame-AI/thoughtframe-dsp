import numpy as np
from thoughtframe.sensor.processors.WindowIsolator import WindowIsolator


class IsolationForestWindowIsolator(WindowIsolator):
    """
    Window isolator driven by ANOMALY RATE slope.

    This defines PHASE WINDOWS (not impulses):

      ENTRY  -> anomaly_rate slope turns positive
      EXIT   -> anomaly_rate slope turns negative

    anomaly_rate is already time-integrated upstream, so:
      - no magnitude thresholds
      - no slope persistence hacks
      - simple latched FSM

    This produces real windows with duration and persists correctly.
    """

    OP_NAME = "if_window_isolator"

    def __init__(self, cfg, sensor):
        super().__init__(cfg, sensor)

        # --------------------------------------------------
        # Config
        # --------------------------------------------------

        # EMA smoothing for anomaly_rate (seconds-scale behavior)
        self.rate_alpha = cfg.get("rate_alpha", 0.05)

        # Deadband to avoid chatter near zero slope
        self.slope_eps = cfg.get("slope_eps", 1e-5)

        # --------------------------------------------------
        # State
        # --------------------------------------------------

        self.prev_t = None

        self.rate_smooth = None
        self.prev_rate_smooth = None

        self.state = "BASELINE"

    @classmethod
    def from_config(cls, cfg, sensor):
        return cls(cfg, sensor)

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------

    def _sign(self, x):
        if x > self.slope_eps:
            return 1
        if x < -self.slope_eps:
            return -1
        return 0

    # --------------------------------------------------
    # FSM
    # --------------------------------------------------

    def propose_state(self, chunk, analysis) -> str:
        m = analysis.metadata
        t = analysis.node.t_sec

        # We only operate if anomaly_rate is present
        if "anomaly_rate" not in m:
            return self.state

        ar = float(m.get("anomaly_rate", 0.0))

        # First sample init
        if self.prev_t is None:
            self.prev_t = t
            self.rate_smooth = ar
            self.prev_rate_smooth = ar
            return self.state

        dt = max(t - self.prev_t, 1e-6)
        self.prev_t = t

        # ----------------------------------------------
        # Smooth anomaly_rate
        # ----------------------------------------------

        self.rate_smooth += self.rate_alpha * (ar - self.rate_smooth)

        # ----------------------------------------------
        # Slope of smoothed anomaly_rate
        # ----------------------------------------------

        slope = (self.rate_smooth - self.prev_rate_smooth) / dt
        self.prev_rate_smooth = self.rate_smooth

        slope_sign = self._sign(slope)

        # ----------------------------------------------
        # Telemetry (for forensics / plots)
        # ----------------------------------------------

        m["anomaly_rate_smooth"] = float(self.rate_smooth)
        m["anomaly_rate_slope"] = float(slope)
        m["anomaly_rate_slope_sign"] = int(slope_sign)

       
       

        # ----------------------------------------------
        # LATCHED STATE MACHINE
        # ----------------------------------------------

        if self.state == "BASELINE":
            # ENTER window on rising anomaly rate
            if slope_sign > 0:
                return "EVENT"

        else:  # EVENT
            # EXIT window on falling anomaly rate
            if slope_sign < 0:
                return "BASELINE"

        return self.state

    # --------------------------------------------------
    # Window stats
    # --------------------------------------------------

    def _init_window_stats(self) -> dict:
        return {
            "anomaly_rate_start": 0.0,
            "anomaly_rate_end": 0.0,
            "anomaly_rate_peak": 0.0,
        }

    def _update_window_stats(self, stats: dict, chunk, analysis) -> None:
        ar = float(analysis.metadata.get("anomaly_rate", 0.0))

        if stats["num_chunks"] == 1:
            stats["anomaly_rate_start"] = ar
            stats["anomaly_rate_peak"] = ar
        else:
            stats["anomaly_rate_peak"] = max(
                stats["anomaly_rate_peak"], ar
            )

        stats["anomaly_rate_end"] = ar
