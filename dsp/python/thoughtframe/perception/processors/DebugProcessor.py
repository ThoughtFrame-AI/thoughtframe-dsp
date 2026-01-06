# thoughtframe/perception/processors/DebugProbeProcessor.py

from thoughtframe.perception.interface import PerceptionProcessor, PerceptionAnalysis


class DebugProcessor(PerceptionProcessor):
    OP_NAME = "debug"

    def process(self, analysis: PerceptionAnalysis) -> None:
        if analysis.tensor is None:
            analysis.metadata["debug"] = "no tensor yet"
            return

        analysis.metadata["shape"] = analysis.tensor.shape
        analysis.metadata["dtype"] = str(analysis.tensor.dtype)

    @classmethod
    def from_config(cls, cfg, source):
        return cls()
