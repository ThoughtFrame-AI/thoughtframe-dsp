from thoughtframe.sensor.processors.WindowClassifier import WindowClassifier


class WindowCnnClassifier(WindowClassifier):
    OP_NAME = "cnn_classifier"

    def classify_window(self, window, analysis):
        # stub for now
        return {
            "type": "cnn_classification_stub",
            "window_id": window.get("id"),
        }

    @classmethod
    def from_config(cls, cfg, sensor):
        return cls()
