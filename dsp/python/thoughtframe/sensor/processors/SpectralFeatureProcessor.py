import numpy as np

from thoughtframe.sensor.interface import AcousticChunkProcessor, \
    AcousticAnalysis

class SpectralFeatureProcessor(AcousticChunkProcessor):
    OP_NAME = "spectral_features"

    def __init__(self, fs: int):
        self.fs = fs

    @classmethod
    def from_config(cls, cfg, sensor):
        return cls(fs=sensor.fs)

    def process(self, chunk: np.ndarray, analysis: AcousticAnalysis) -> None:
        # time-domain
        rms = float(np.sqrt(np.mean(chunk ** 2)))

        # frequency-domain (lazy FFT)
        fft = analysis.fft
        freqs = analysis.fft_freqs
        mag = np.abs(fft)
        power = mag ** 2

        centroid = np.sum(freqs * power) / np.sum(power)
        bandwidth = np.sqrt(
            np.sum(((freqs - centroid) ** 2) * power) / np.sum(power)
        )
        rolloff = freqs[np.searchsorted(np.cumsum(power), 0.85 * np.sum(power))]
        flatness = np.exp(np.mean(np.log(power + 1e-12))) / np.mean(power)

        # time–frequency (lazy STFT → mel → log-mel)
        log_mel = analysis.log_mel
        mel_mean = float(log_mel.mean())
        mel_std = float(log_mel.std())

        analysis.metadata.update({
            "rms": rms,
            "spec_centroid_hz": float(centroid),
            "spec_bandwidth_hz": float(bandwidth),
            "spec_rolloff_hz": float(rolloff),
            "spec_flatness": float(flatness),
            "mel_mean_energy": mel_mean,
            "mel_std_energy": mel_std,
        })
