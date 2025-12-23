from thoughtframe.sensor.interface import AcousticChunkProcessor


class DebugProbeProcessor(AcousticChunkProcessor):
    
    def process(self, chunk, analysis):
        analysis.metadata["chunk_len"] = len(chunk)
        
    @classmethod
    def from_config(cls, cfg, sensor):
        return cls()