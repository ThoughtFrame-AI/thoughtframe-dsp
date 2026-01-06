# thoughtframe/ml/acoustic_worker.py

import numpy as np
import librosa
from sklearn.ensemble import IsolationForest
from thoughtframe.sensor.interface import AcousticChunkProcessor
from thoughtframe.sensor.interface import AcousticAnalysis

class IsolationForestProcessor(AcousticChunkProcessor):
    
    OP_NAME = "isolation_forest" 

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
        threshold = cfg.get("threshold", -0.5)        
        return cls(fs, threshold)

   

    def process(self, chunk: np.ndarray, analysis: AcousticAnalysis) -> None:
        
        log_mel = analysis.log_mel

        mean = log_mel.mean(axis=1)
        std  = log_mel.std(axis=1)

        feature_vector = np.concatenate([mean, std])
        
        if not self._trained:
            self._feature_buffer.append(feature_vector)
            if len(self._feature_buffer) >= 2000:
                self.detector.fit(self._feature_buffer)
                self._trained = True
                print("[ml] detector trained")
            return None

        score = self.detector.decision_function([feature_vector])[0]
        analysis.metadata["iforest_score"] = float(score)    
            
            
         
