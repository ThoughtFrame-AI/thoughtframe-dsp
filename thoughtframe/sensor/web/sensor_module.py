from tf_core.modules.BaseFrameModule import BaseFrameModule
from tf_core.bootstrap import configure, thoughtframe

from thoughtframe.sensor.utils import spectrogram


class SensorModule(BaseFrameModule):
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
        self.mesh  = thoughtframe.get("sensormeshmanager")

   
    def run_test_command(self, request):
        print(f"Executing test command {request}")
        
    def status(self, request):
        return {
            "module": self.__class__.__name__,
            "status": "ready"
        }
    
    def startRun(self, request):
        return self.mesh.start(request)
            
        
    def generate_spectrogram(self, request):
        """
        Input: { 
            "audio_url": "http://...", 
            "output_path": "/tmp/spec.png",
            "n_fft": 4096 
        }
        """
        print(f"[SensorModule] Generating Spectrogram for {request.get('audio_url')}")
        
        try:
            # Delegate to your library
            # 'request' is the dictionary containing audio_url, output_path, etc.
            saved_path = spectrogram.generate_spectrogram(request)
            
            return {
                "status": "success",
                "output_path": saved_path
            }
        except Exception as e:
            print(f"Spectrogram failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }