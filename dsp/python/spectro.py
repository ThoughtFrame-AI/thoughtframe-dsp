import matplotlib.pyplot as plt
import soundfile as sf
import numpy as np

# The file containing the "Windchimes" event
FILE_PATH = "/home/ian/git/thoughtframe-dsp/audio/beam_0/timewindow_beam_0_506306_509821.flac"

# Forensic Settings for Modem Detection
TARGET_OFFSET_SEC = 55.0   # Where we saw the waveform "fuzz" in the macro view
WINDOW_SIZE_SEC = 0.5      # Zoom in tight (500ms) to see the individual bits
NFFT = 256                 # Low NFFT = High Time Resolution (crucial for chirps)

def plot_modem_zoom():
    print(f"Loading {FILE_PATH}...")
    try:
        # Read file metadata
        info = sf.info(FILE_PATH)
        fs = info.samplerate
        
        # Calculate sample indices
        start_sample = int((TARGET_OFFSET_SEC - (WINDOW_SIZE_SEC/2)) * fs)
        stop_sample = int((TARGET_OFFSET_SEC + (WINDOW_SIZE_SEC/2)) * fs)
        
        # Read the slice
        data, fs = sf.read(FILE_PATH, start=start_sample, stop=stop_sample)
        
        # Mix to mono if needed
        if len(data.shape) > 1:
            data = data.mean(axis=1)

        # PLOT
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [1, 2]})
        
        # 1. Waveform (The Packets)
        t = np.linspace(0, len(data)/fs, len(data))
        ax1.plot(t, data, color='#2c3e50', linewidth=1)
        ax1.set_title(f"Waveform: High-Speed Transient (t={TARGET_OFFSET_SEC}s)", fontsize=14)
        ax1.set_ylabel("Amplitude")
        ax1.grid(True, alpha=0.5)
        
        # 2. Spectrogram (The Digital Handshake)
        # Low NFFT (256) smears frequency slightly but gives us razor-sharp time boundaries
        Pxx, freqs, bins, im = ax2.specgram(data, NFFT=NFFT, Fs=fs, noverlap=NFFT//2, cmap='inferno')
        
        ax2.set_title("Spectral Logic: Searching for FSK/Chirps", fontsize=14)
        ax2.set_ylabel("Frequency (Hz)")
        ax2.set_xlabel("Time (seconds)")
        
        # Add a colorbar
        plt.colorbar(im, ax=ax2, label='Intensity (dB)')
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    plot_modem_zoom()