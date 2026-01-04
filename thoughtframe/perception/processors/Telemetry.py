# thoughtframe/perception/processors/DebugProbeProcessor.py

from thoughtframe.perception.interface import PerceptionProcessor, PerceptionAnalysis


class TelemetryProcessor(PerceptionProcessor):
    OP_NAME = "telemetry"

    def process(self, analysis: PerceptionAnalysis) -> None:
        if analysis.tensor is None:
            analysis.metadata["debug"] = "no tensor yet"
            return

    
    @classmethod
    def from_config(cls, cfg, source):
        return cls()
