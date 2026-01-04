from abc import ABC, abstractmethod

from thoughtframe.sensor.interface import AcousticChunkProcessor

class WindowClassifier(AcousticChunkProcessor, ABC):
    """
    Operates only on finalized windows.
    Never influences window formation.
    """

    def process(self, chunk, analysis):
        if not self._is_window_close(analysis):
            return

        window = analysis.metadata.get("window")
        if window is None:
            return

        result = self.classify_window(window, analysis)

        if result is not None:
            self.emit(result, window, analysis)

    @abstractmethod
    def classify_window(self, window: dict, analysis):
        pass
