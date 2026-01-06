from thoughtframe.sensor.processors.WindowClassifier import WindowClassifier
import numpy as np


class WindowClusterClassifier(WindowClassifier):
    OP_NAME = "window_cluster"

    def __init__(self, centroids=None, threshold=0.5):
        self.centroids = centroids or []
        self.threshold = threshold

    def classify_window(self, window, analysis):
        embedding = analysis.metadata.get("window_embedding")
        if embedding is None:
            return None

        x = np.asarray(embedding, dtype=np.float32)

        if not self.centroids:
            return {
                "type": "window_cluster",
                "cluster_id": None,
                "novel": True,
                "distance": None,
            }

        distances = [np.linalg.norm(x - c) for c in self.centroids]
        best = int(np.argmin(distances))
        d = float(distances[best])

        return {
            "type": "window_cluster",
            "cluster_id": best,
            "distance": d,
            "novel": d > self.threshold,
        }

    @classmethod
    def from_config(cls, cfg, sensor):
        return cls(
            centroids=cfg.get("centroids"),
            threshold=cfg.get("threshold", 0.5),
        )
