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
                "-i", "samples/MARS-20150730T000000Z-16kHz.wav",
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
                {"op": "ring_buffer", "seconds": 20},
                {"op": "snapshot"},

                
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
