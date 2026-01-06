from tf_core.modules.BaseFrameModule import BaseFrameModule
from tf_core.bootstrap import configure, thoughtframe
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from thoughtframe.sensor.utils import spectrogram
from tf_core.bootstrap import thoughtframe


class PerceptionModule(BaseFrameModule):
    """
    Python-side equivalent of a ThoughtFrame module.
    Handles:
      - calling TF runpaths over HTTP
      - emitting events back into the mesh
      - generic frame-level utilities
      - simple serialization helpers
    """
    def __init__(self):
        super().__init__()
        print("Initialized")
        self.mesh  = thoughtframe.get("perceptionmeshmanager")
   
    def run_test_command(self, request):
        print(f"Executing test command {request}")
        
    def status(self, request):
        return {
            "module": self.__class__.__name__,
            "status": "ready"
        }
        
    def process(self, request):
        # Control-plane only: validate, enqueue, return
        
        manager = self.mesh
        manager.start()

        mesh = manager.mesh   
        source_id = request.get("source")
        if not source_id:
            source_id = "default"

        source = mesh.sources.get(source_id)
        
        count = 0
        for item in request["items"]:
            source.push(item)
            count += 1
    
        return {
            "status": "enqueued",
            "count": count
        }    
        
        
    
