# thoughtframe/ml/acoustic_worker.py

import numpy as np
import librosa
from sklearn.ensemble import IsolationForest
from thoughtframe.sensor.interface import AcousticChunkProcessor
from thoughtframe.sensor.interface import AcousticAnalysis

class IsolationForestProcessor(AcousticChunkProcessor):
    def __init__(self, fs, threshold):
        self.fs = fs
        self.threshold = threshold
        self.detector = IsolationForest(
            n_estimators=100,
            contamination=0.05,
            random_state=42
        )
        self._trained = False
        self._feature_buffer = []

    @classmethod    
    def from_config(cls, cfg, sensor):
        fs = sensor.fs
        threshold = cfg.get("threshold", 0.15)        
        return cls(fs, threshold)

    def extract_features(self, chunk):
        print("rms:", np.sqrt(np.mean(chunk**2)))

        ## This is taking a long vector, chunk, of audio
        ##doing FFTs of width 512 and sliding the window
        ##256 to the right
        ##result is complex arrays of length 512 
        S = librosa.stft(chunk, n_fft=512, hop_length=256)
        ##This removes the phase and converts
        ##back to real numbers representing the power
        ##(length of the complex vector)
        S_mag = np.abs(S)

        ##this reduces the vector to only 40
        ##focusing on lower frequencies - not sure
        ##what all the params are..
        ## so we have many rows of width/length 40?
        mel = librosa.feature.melspectrogram(
            S=S_mag**2,
            sr=self.fs,
            n_mels=40,
            fmin=50,
            fmax=self.fs / 2
        )
        ## convert to DB 
        log_mel = librosa.power_to_db(mel)
        ##create a new vector of the mean of            
        mean_vector = log_mel.mean(axis=1)
        ##create a new vector of the standard deviation of eacb
        std_vector = log_mel.std(axis=1)
       ## print(mean_vector, std_vector)

        return np.concatenate([
            mean_vector,
            std_vector
        ])

    def process(self, chunk: np.ndarray, analysis: AcousticAnalysis) -> None:
        feat = self.extract_features(chunk)
        
        if not self._trained:
            self._feature_buffer.append(feat)
            if len(self._feature_buffer) >= 50:
                self.detector.fit(self._feature_buffer)
                self._trained = True
                print("[ml] detector trained")
            return None

        score = self.detector.decision_function([feat])[0]
        if score < self.threshold:
            print("Anomoly detected")
            analysis.flags.add("saverequested")
            analysis.flags.add("anomolydetected")
            print(analysis.flags)
            
            
        return 
