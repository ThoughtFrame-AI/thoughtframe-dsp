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
from thoughtframe.sensor.mesh_config import THOUGHTFRAME_CONFIG


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
        audio_url = request.get("audio_url")
        rel_output = request.get("output_path")
        sensor_id  = request.get("beam_id")   
    
        if not audio_url or not rel_output or not sensor_id:
            return {
                "status": "error",
                "error": "Missing audio_url, output_path, or beam_id"
            }
    
        # EXACT same root logic as ForensicSummaryProcessor
        data_root = thoughtframe.resolve_rooted_path(
            THOUGHTFRAME_CONFIG,
            THOUGHTFRAME_CONFIG.get("samples", "audio"),
            sensor_id
        )
    
        abs_output = os.path.join(data_root, rel_output.lstrip("/"))
        outdir = os.path.dirname(abs_output)
        os.makedirs(outdir, exist_ok=True)
    
        if os.path.exists(abs_output):
            return {
                "status": "cached",
                "output_path": abs_output
            }
    
        try:
            req = dict(request)
            req["output_path"] = abs_output
    
            saved_path = spectrogram.generate_spectrogram(req)
    
            return {
                "status": "success",
                "output_path": saved_path
            }
    
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }


    def cluster_windows(self, request):
        try:
            windows = request.get("windows", [])
            k = int(request.get("k", 3))
            make_plots = bool(request.get("make_plots", False))
            rel_plot_dir = request.get("plot_dir")
            sensor_id = request.get("beam_id")
    
            if not windows or not sensor_id:
                return {
                    "status": "error",
                    "error": "Missing windows or beam_id"
                }
    
            FEATURE_COLS = [
                "rms_mean",
                "rms_var",
                "centroid_mean",
                "centroid_var",
            ]
    
            ids, X, start_t = [], [], []
    
            for w in windows:
                try:
                    X.append([float(w[c]) for c in FEATURE_COLS])
                    ids.append(w["window_id"])
                    start_t.append(float(w["start_t"]))
                except Exception:
                    continue
    
            if len(X) < k:
                return {
                    "status": "error",
                    "error": "Not enough valid windows for k-means"
                }
    
            X = np.array(X)
            Xs = StandardScaler().fit_transform(X)
    
            km = KMeans(n_clusters=k, n_init="auto", random_state=42)
            labels = km.fit_predict(Xs)
    
            clusters = {ids[i]: int(labels[i]) for i in range(len(ids))}
    
            result = {
                "status": "success",
                "clusters": clusters,
            }
    
            # --------------------------------------------------
            # OPTIONAL PLOTS (PATH FIXED)
            # --------------------------------------------------
            if make_plots and rel_plot_dir:
                data_root = thoughtframe.resolve_rooted_path(
                    THOUGHTFRAME_CONFIG,
                    THOUGHTFRAME_CONFIG.get("samples", "audio"),
                    sensor_id
                )
    
                plot_dir = os.path.join(data_root, rel_plot_dir.lstrip("/"))
                os.makedirs(plot_dir, exist_ok=True)
    
                f1 = os.path.join(plot_dir, "kmeans_features.png")
                f2 = os.path.join(plot_dir, "kmeans_time.png")
    
                # Cache guard
                if not (os.path.exists(f1) and os.path.exists(f2)):
                    matplotlib.use("Agg")
    
                    # Feature space
                    plt.figure(figsize=(6, 5))
                    sc = plt.scatter(
                        X[:, 0],
                        X[:, 2],
                        c=labels,
                        cmap="tab10",
                        s=40
                    )
                    plt.xlabel("RMS mean")
                    plt.ylabel("Centroid mean")
                    plt.title("KMeans – Feature Space")
                    plt.colorbar(sc, label="Cluster")
                    plt.tight_layout()
                    plt.savefig(f1, dpi=150)
                    plt.close()
    
                    # Time order
                    plt.figure(figsize=(10, 3))
                    plt.scatter(start_t, labels, c=labels, cmap="tab10", s=30)
                    plt.xlabel("Time (s)")
                    plt.ylabel("Cluster")
                    plt.title("KMeans – Cluster vs Time")
                    plt.yticks(range(k))
                    plt.tight_layout()
                    plt.savefig(f2, dpi=150)
                    plt.close()
    
                result["plots"] = {
                    "feature_space": f1,
                    "time": f2
                }
    
            return result
    
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }


