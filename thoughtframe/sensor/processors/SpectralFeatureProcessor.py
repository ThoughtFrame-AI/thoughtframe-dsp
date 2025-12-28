# thoughtframe/sensor/processors/ring_buffer.py

import time
import threading
from collections import deque
from queue import Queue
import numpy as np
import soundfile as sf


from thoughtframe.sensor.interface import AcousticChunkProcessor, AcousticAnalysis


class SpectralFeatureProcessor(AcousticChunkProcessor):
    """
    Rolling in-memory audio buffer with asynchronous snapshot persistence.
    """

    def __init__(self,fs: int):
        self.fs = fs
        

    @classmethod
    def from_config(cls, cfg, sensor):
        return cls(
            fs=sensor.fs
        )

    def process(self, chunk: np.ndarray, analysis: AcousticAnalysis) -> None:
        rms = np.sqrt(np.mean(chunk ** 2))

        fft = analysis.fft
        freqs = analysis.fft_freqs
        
        ##fft incudes phase so remove 
        mag = np.abs(fft)
        ##turn to power
        power = mag ** 2
        ##what are freqs speciically?  
        
        ##find a peak?
        centroid = np.sum(freqs * power) / np.sum(power)
        ##width of the peak? 
        bandwidth = np.sqrt(np.sum(((freqs - centroid) ** 2) * power) / np.sum(power))
        rolloff = freqs[np.searchsorted(np.cumsum(power), 0.85 * np.sum(power))]
        flatness = np.exp(np.mean(np.log(power + 1e-12))) / np.mean(power)
        
        analysis.metadata.update({
        "rms": float(rms),
        "spec_centroid_hz": float(centroid),
        "spec_bandwidth_hz": float(bandwidth),
        "spec_rolloff_hz": float(rolloff),
        "spec_flatness": float(flatness),
        
        })
        