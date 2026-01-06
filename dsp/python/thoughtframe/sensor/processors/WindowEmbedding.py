from thoughtframe.sensor.processors.WindowClassifier import WindowClassifier


class WindowEmbedding(WindowClassifier):
    """
    Projects finalized window statistics into an embedding vector.
    Uses isolator-produced data only.
    """
    OP_NAME = "window_embedding"

    def classify_window(self, window, analysis):
        # Select a stable subset of isolator stats
        embedding = [
            window.get("rms_mean"),
            window.get("rms_var"),
            window.get("centroid_mean"),
            window.get("centroid_var"),
            window.get("duration_sec"),
            window.get("num_chunks"),
        ]

        # Defensive: drop None values but keep order
        embedding = [v if v is not None else 0.0 for v in embedding]

        analysis.metadata["window_embedding"] = embedding

        return {
            "type": "window_embedding",
            "window_id": window.get("window_id") or window.get("id"),
            "dim": len(embedding),
        }

    @classmethod
    def from_config(cls, cfg, sensor):
        return cls()
