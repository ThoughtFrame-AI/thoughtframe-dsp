import subprocess
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# ==========================================
# CONFIGURATION
# ==========================================
# The exact URL you just tested
BASE_URL = "http://localhost:8080/thoughtframe/lab/dsp/audio/boat_event.flac"
PARAMS = "?beam=beam_0&start_s=7000&end_s=7500"
FULL_URL = BASE_URL + PARAMS

SAMPLE_RATE = 48000
N_FFT = 4096        # High resolution for frequency
HOP_LENGTH = 1024   # Overlap

def stream_audio_from_url(url):
    print(f"Streaming from: {url}")
    
    # 1. Use FFmpeg to stream the URL and convert to raw float32 on the fly
    # This handles the FLAC decoding transparently
    command = [
        "ffmpeg",
        "-i", url,
        "-f", "f32le",       # Convert to raw float 32-bit
        "-ac", "1",          # Force mono
        "-ar", str(SAMPLE_RATE),
        "-loglevel", "error",
        "-"                  # Output to stdout
    ]

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    # 2. Read the raw bytes from stdout
    raw_audio, stderr = process.communicate()

    if process.returncode != 0:
        print("FFmpeg Error:", stderr.decode())
        return None

    # 3. Convert bytes to numpy array
    audio_data = np.frombuffer(raw_audio, dtype=np.float32)
    print(f"Received {len(audio_data)} samples ({len(audio_data)/SAMPLE_RATE:.2f} seconds)")
    return audio_data

def plot_spectrogram(audio):
    plt.figure(figsize=(16, 8))
    
    # Use matplotlib's specgram
    # NFFT: Window size
    # Fs: Sample rate (for axis labels)
    # noverlap: Overlap size
    Pxx, freqs, bins, im = plt.specgram(
        audio, 
        NFFT=N_FFT, 
        Fs=SAMPLE_RATE, 
        noverlap=N_FFT - HOP_LENGTH,
        cmap='inferno',       # 'inferno' or 'magma' looks great for audio
        mode='magnitude',
        scale='dB'            # Show in Decibels
    )

    plt.title(f"Spectrogram: Boat Event (48kHz High-Def)\n{FULL_URL}")
    plt.ylabel("Frequency (Hz)")
    plt.xlabel("Time (Seconds)")
    
    # Limit Y-axis if you want to focus on the engine (0-5kHz) 
    # or leave it open to see the full 24kHz bandwidth
    # plt.ylim(0, 5000) 

    plt.colorbar(im, format='%+2.0f dB').set_label('Intensity (dB)')
    plt.tight_layout()
    plt.show()

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    audio = stream_audio_from_url(FULL_URL)
    if audio is not None and len(audio) > 0:
        plot_spectrogram(audio)
    else:
        print("No audio data received.")