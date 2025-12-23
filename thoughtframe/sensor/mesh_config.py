# thoughtframe/sensor/sensormesh_config.py

MESH_CONFIG = {
    "sensors": [
        {
            "type": "ffmpeg",
            "id": "mic1",
            "cmdX": [
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
                "-re",                
                "-i", "/tmp/audio_snapshot_mic1_1766426618.wav",
                "-ac", "1",
                "-ar", "8000",
                "-f", "f32le",
                "-"
            ],
            "fs": 8000,
            "chunk_size": 1024,
            "pipeline": [
                {"op": "debug"},
                {"op": "isolation_forest", "threshold": 0.05},
                ##{"op": "ring_buffer", "seconds": 20},

                
            ]
        }
    ]
}


NETWORK_CONFIG  = {
    "baseurl":"https://localhost:8080",
    "websocket":"ws://localhost:8080/entermedia/services/websocket/org/thoughtframe/websocket/BenchConnection?sessionid={tabID}",
    "apphome":"/thoughtframe"
    
    
    
}
