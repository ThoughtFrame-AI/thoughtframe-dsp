from abc import ABC, abstractmethod
from typing import AsyncIterator

import librosa

import numpy as np


class AcousticAnalysis:
    def __init__(self, chunk, node, timestamp=None):
        self.chunk = chunk
        self.node = node
        self.sensor_id = node.sensor.sensor_id
        self.timestamp = timestamp
        self.events: list[dict] = []
        self.metadata: dict = {}
        self.flags: set[str] = set()   
        self._fft = None
        self._fft_freqs = None
        self._stft = None
        self._mel = None
        self._log_mel = None
        
    @property
    def chunk_index(self):
        return self.node.chunk_index
    
    @property
    def fft(self):
        ## TODO:  TEst with hanning windows also
        # # 1. Create the Hanning Window
        #     # (In production, pre-calculate this once in __init__ for speed)
        #     window = np.hanning(len(self.chunk))
        #
        #     # 2. Apply Window -> Compute FFT
        #     self._fft = np.fft.rfft(self.chunk * window)
        #
        #     self._fft_freqs = np.fft.rfftfreq(
        #         len(self.chunk),
        #         d=1.0 / self.node.sensor.fs
        #     )
        #

        if self._fft is None:
            self._fft = np.fft.rfft(self.chunk)
            self._fft_freqs = np.fft.rfftfreq(
                len(self.chunk),
                d=1.0 / self.node.sensor.fs
            )
        return self._fft

    @property
    def fft_freqs(self):
        # ensure fft computed
        _ = self.fft
        return self._fft_freqs
    
    @property
    def stft(self):
        if self._stft is None:
            self._stft = librosa.stft(
                self.chunk,
                n_fft=512,
                hop_length=256
            )
        return self._stft
    
    @property
    def mel(self):
        if self._mel is None:
            S_mag = np.abs(self.stft) ** 2
            self._mel = librosa.feature.melspectrogram(
                S=S_mag,
                sr=self.node.sensor.fs,
                n_mels=40,
                fmin=50,
                fmax=self.node.sensor.fs / 2
            )
        return self._mel

    @property
    def log_mel(self):
        if self._log_mel is None:
            self._log_mel = librosa.power_to_db(self.mel)
        return self._log_mel
    
    @property
    def complex_tensor(self):
        """
        Returns a stacked tensor of shape (2 * N_channels, Freq, Time).
        For mono: (2, F, T) -> [Real, Imag]
        """
        # Get the complex STFT (already computed in self.stft via librosa)
        z = self.stft  # Shape (F, T), dtype=complex128 or complex64
        
        # We need to handle potential multi-channel inputs in the future.
        # If z is 2D (Freq, Time), we treat it as 1 channel.
        if z.ndim == 2:
            # Stack Real and Imaginary parts: Shape (2, F, T)
            return np.stack([z.real, z.imag], axis=0)
        
        # If z is 3D (Channels, Freq, Time) for your future array:
        elif z.ndim == 3:
            # Result: (2 * Channels, F, T)
            return np.concatenate([z.real, z.imag], axis=0)

class AcousticChunkProcessor(ABC):
    OP_NAME: str | None = None

    @abstractmethod
    def process(self, chunk: np.ndarray, analysis: AcousticAnalysis) -> None:
        pass
    @classmethod
    @abstractmethod
    def from_config(cls, cfg, sensor):
        raise NotImplementedError(
            f"{cls.__name__}.from_config() must be implemented"
        )



class AcousticSensor(ABC):
    def __init__(self, sensor_id: str,fs: int, chunk_size: int):
        self.sensor_id = sensor_id
        self.fs = fs
        self.chunk_size = chunk_size

    @abstractmethod
    async def stream(self) -> AsyncIterator[np.ndarray]:
        """
        Async generator yielding np.float32 arrays of shape (chunk_size,)
        """
        pass
    
    @classmethod
    @abstractmethod
    def from_config(cls, cfg):
        raise NotImplementedError(
            f"{cls.__name__}.from_config() must be implemented"
        )

class AcousticPipeline:
    
    def __init__(self):
        self.filters: list[AcousticChunkProcessor] = []
    
    def addChunkProcessor(self, chunkprocessor: AcousticChunkProcessor):
        self.filters.append(chunkprocessor)

    def execute(self, chunk,node):
        analysis :AcousticAnalysis = AcousticAnalysis(chunk,node)
        #print(analysis.flags)
        for processor in self.filters:
            
            processor.process(chunk,analysis)
        return analysis


