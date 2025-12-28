# thoughtframe/sensor/sensormesh_config.py

MESH_CONFIG = {
    "sensors": [
        {
            "type": "ffmpeg",
            "id": "mic1",
            "cmd-mic": [
                "ffmpeg",
                "-f", "alsa",
                "-i", "default",
                "-ac", "1",
                "-ar", "8000",
                "-f", "f32le",
                "-"
            ],
            "cmd": [
                "ffmpeg",
                "-i", "samples/OneDayHydrophone.flac",
                "-ac", "1",
                "-ar", "8000",
                "-f", "f32le",
                "-"
            ],
            "fs": 8000,
            "chunk_size": 1024,
            "pipeline": [
                {"op": "debug"},
                {"op": "spectral_features"},
                {"op": "isolation_forest", "threshold": -0.1},
                {"op": "temporal_context", "time": "1h"},
                #{"op": "window_isolator", "enter_threshold": 0.08, "exit_threshold": 0.05, "min_duration": "15s"},
                {"op": "if_window_isolator", "threshold" : -0.1, "min_duration": "5m"},
                {"op": "time_window_isolator", "enter_threshold": 0.08, "width": "2m", "min_duration": "15s"},
                {"op": "guard",
                 "name": "impulses_high",
                 "band": [600, 3000],
                 "pipeline": [
                     {"op": "impulse_isolator",
                      "threshold_mult": 10.0,
                      "window_sec": 5.0,
                      "min_impulses": 5}
                 ]
                },
                {"op": "guard",
                 "name": "impulses_low",
                 "band": [20, 250],
                 "pipeline": [
                     {"op": "impulse_isolator",
                      "threshold_mult": 10.0,
                      "window_sec": 5.0,
                      "min_impulses": 5}
                 ]
                },

 
                
               
                ##{"op": "ring_buffer", "seconds": 20},
                ##{"op": "snapshot"},
                {"op": "telemetry"},

                
            ]
        }
    ]
}


THOUGHTFRAME_CONFIG  = {
    "root":".",
    "samples":"/audio"
}




NETWORK_CONFIG  = {
    "baseurl":"https://localhost:8080",
    "websocket":"ws://localhost:8080/entermedia/services/websocket/org/thoughtframe/websocket/BenchConnection?sessionid={tabID}",
    "apphome":"/thoughtframe"
}
