import numpy as np
from thoughtframe.sensor.processors.WindowIsolator import WindowIsolator


class DcmtIsolator(WindowIsolator):
    """
    DCMT pin-based isolator.

    Pins are learned centroids in embedding space.
    Windows correspond to time spent near a pin.
    """

    OP_NAME = "dcmt_isolator"

    def __init__(self, cfg, sensor):
        super().__init__(cfg, sensor)

        # geometry
        self.pin_radius     = cfg.get("pin_radius", 0.01)
        self.new_pin_radius = cfg.get("new_pin_radius", 0.03)

        # centroid update speed
        self.pin_alpha = cfg.get("pin_alpha", 0.02)

        # persistence before spawning a new pin
        self.spawn_chunks = cfg.get("spawn_chunks", 8)

        self.pins = []                 # list[np.ndarray]
        self.active_pin = None
        self._far_count = 0

    @classmethod
    def from_config(cls, cfg, sensor):
        return cls(cfg, sensor)

    # ------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------

    def _dist(self, a, b):
        return float(np.linalg.norm(a - b))

    def _find_nearest_pin(self, emb):
        best_i = None
        best_d = None
        for i, p in enumerate(self.pins):
            d = self._dist(emb, p)
            if best_d is None or d < best_d:
                best_d = d
                best_i = i
        return best_i, best_d

    # ------------------------------------------------------------
    # FSM driver
    # ------------------------------------------------------------

    def propose_state(self, chunk, analysis) -> str:
        emb = getattr(analysis, "_dcmt_embedding", None)
        if emb is None:
            return self.state or "BASELINE"

        # First embedding → seed first pin
        if not self.pins:
            self.pins.append(emb.copy())
            self.active_pin = 0
            self._far_count = 0
            self._write_metadata(analysis, 0, 0.0)
            print("[DCMT] created PIN_0")
            return "PIN_0"

        # distance to active pin
        active_centroid = self.pins[self.active_pin]
        d_active = self._dist(emb, active_centroid)

        # --------------------------------------------------------
        # Case 1: still inside active pin → update centroid
        # --------------------------------------------------------
        if d_active <= self.pin_radius:
            self._far_count = 0
            self.pins[self.active_pin] += self.pin_alpha * (emb - active_centroid)
            self._write_metadata(analysis, self.active_pin, d_active)
            return f"PIN_{self.active_pin}"

        # --------------------------------------------------------
        # Case 2: closer to another existing pin
        # --------------------------------------------------------
        nearest, d_nearest = self._find_nearest_pin(emb)
        if nearest is not None and d_nearest <= self.pin_radius:
            self.active_pin = nearest
            self._far_count = 0
            self.pins[nearest] += self.pin_alpha * (emb - self.pins[nearest])
            self._write_metadata(analysis, nearest, d_nearest)
            print(f"[DCMT] switched to PIN_{nearest}")
            return f"PIN_{nearest}"

        # --------------------------------------------------------
        # Case 3: far from everything → possible new pin
        # --------------------------------------------------------
        self._far_count += 1

        if d_active >= self.new_pin_radius and self._far_count >= self.spawn_chunks:
            self.pins.append(emb.copy())
            self.active_pin = len(self.pins) - 1
            self._far_count = 0
            self._write_metadata(analysis, self.active_pin, 0.0)
            print(f"[DCMT] spawned PIN_{self.active_pin}")
            return f"PIN_{self.active_pin}"

        # --------------------------------------------------------
        # Case 4: liminal drift → stay but do not update centroid
        # --------------------------------------------------------
        self._write_metadata(analysis, self.active_pin, d_active)
        return f"PIN_{self.active_pin}"

    # ------------------------------------------------------------
    # Metadata + window stats
    # ------------------------------------------------------------

    def _write_metadata(self, analysis, pin_id, dist):
        analysis.metadata["dcmt_pin_id"] = pin_id
        analysis.metadata["dcmt_pin_dist"] = dist

    def _init_window_stats(self):
        return {
            "dcmt_pin_id": None,
            "mean_pin_dist": 0.0,
        }

    def _update_window_stats(self, stats, chunk, analysis):
        pid = analysis.metadata.get("dcmt_pin_id")
        dist = analysis.metadata.get("dcmt_pin_dist")

        stats["dcmt_pin_id"] = pid

        if dist is None:
            return

        n = stats["num_chunks"]
        if n == 1:
            stats["mean_pin_dist"] = dist
        else:
            m = stats["mean_pin_dist"]
            stats["mean_pin_dist"] = m + (dist - m) / n
