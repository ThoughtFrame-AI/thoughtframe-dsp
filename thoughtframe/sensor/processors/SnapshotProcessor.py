from thoughtframe.sensor.interface import AcousticChunkProcessor


class SnapshotProcessor(AcousticChunkProcessor):
    def process(self, chunk, analysis):
        analysis.metadata["chunk_len"] = len(chunk)