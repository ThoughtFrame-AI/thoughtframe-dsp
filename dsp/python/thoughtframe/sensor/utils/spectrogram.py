import subprocess
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt


def _fetch_audio(url: str, sample_rate: int) -> np.ndarray:
    """
    Decode audio from a URL into mono float32 samples using ffmpeg.
    """
    cmd = [
        "ffmpeg",
        "-i", url,
        "-f", "f32le",
        "-ac", "1",
        "-ar", str(sample_rate),
        "-loglevel", "error",
        "-"
    ]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    raw, err = proc.communicate()

    if proc.returncode != 0:
        raise RuntimeError(err.decode())

    return np.frombuffer(raw, dtype=np.float32)


def _render_spectrogram(
    audio: np.ndarray,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    output_path: Path,
):
    plt.figure(figsize=(12, 6))

    _, _, _, im = plt.specgram(
        audio,
        NFFT=n_fft,
        Fs=sample_rate,
        noverlap=n_fft - hop_length,
        cmap="inferno",
        mode="magnitude",
        scale="dB",
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.colorbar(im, format="%+2.0f dB")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def generate_spectrogram(payload: dict) -> str:
    """
    Pure payload-driven generator.

    Required payload keys:
      - audio_url
      - output_path

    Optional:
      - sample_rate (default 48000)
      - n_fft (default 4096)
      - hop_length (default 1024)

    Returns:
      output_path (str)
    """
    audio_url = payload["audio_url"]
    output_path = Path(payload["output_path"])

    sample_rate = payload.get("sample_rate", 48000)
    n_fft = payload.get("n_fft", 4096)
    hop_length = payload.get("hop_length", 1024)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    audio = _fetch_audio(audio_url, sample_rate)

    if audio.size == 0:
        raise ValueError("Empty audio buffer")

    _render_spectrogram(
        audio=audio,
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        output_path=output_path,
    )

    return str(output_path)
