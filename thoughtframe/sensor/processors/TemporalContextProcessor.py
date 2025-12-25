from thoughtframe.sensor.interface import AcousticChunkProcessor
from pytimeparse.timeparse import timeparse
from numba.core.types import none


class TemporalContextProcessor(AcousticChunkProcessor):
    
    
    def __init__(self, cfg, sensor):
        self.cfg = cfg
        self.sensor = sensor
        
        self.rms_mean = None
        self.rms_var = None
        self.centroid_mean = None
        self.anomaly_rate = 0.0
        
        self.timescale = cfg["time"]        
        self.fs = sensor.fs
        self.chunk_size=sensor.chunk_size   
        self.decayfactor = self.load_decay(self.timescale)
        
    
    def process(self, chunk, analysis):
        chunk_rms:float = analysis.metadata.get("rms")
        
        if chunk_rms is None:
            return
            
        if self.rms_mean is None:
            self.rms_mean = chunk_rms
            self.rms_var = 0.0
        else:
            self.rms_mean = self.rms_mean + self.decayfactor * (chunk_rms - self.rms_mean)

        analysis.metadata["rms_mean"] = self.rms_mean
        mean_delta_squared = (chunk_rms - self.rms_mean) ** 2
        self.rms_var = self.rms_var + self.decayfactor * (mean_delta_squared - self.rms_var)
        analysis.metadata["rms_var"] = self.rms_var

        centroid:float = analysis.metadata.get("spec_centroid_hz")
        if centroid is not None:
            self.centroid_mean = centroid if self.centroid_mean is None else self.centroid_mean
            self.centroid_mean = self.centroid_mean + self.decayfactor * (centroid - self.centroid_mean)
            analysis.metadata["centroid_mean"] = self.centroid_mean

        is_anomaly = 1.0 if "anomolydetected" in analysis.flags else 0.0
        
        self.anomaly_rate = (self.anomaly_rate + self.decayfactor * (is_anomaly - self.anomaly_rate))
        if(self.anomaly_rate > .12):
            print(f"Mean: {self.rms_mean} Variance: {self.rms_var} Anomoly Rate: {self.anomaly_rate}")    
        
    
    
    
        
    @classmethod
    def from_config(cls, cfg, sensor):
        return cls(cfg,sensor)
    
    def load_decay(self, timescale : str):
        dt = self.chunk_size / self.fs
        seconds = timeparse(timescale)
        return dt / seconds;
    