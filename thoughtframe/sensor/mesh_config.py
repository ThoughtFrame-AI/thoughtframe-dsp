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
                "-i", "samples/MARS-20150815T000000Z-2kHz.wav",
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
                {"op": "ring_buffer", "seconds": 20},
                {"op": "snapshot"},

                
            ]
        }
    ]
}


THOUGHTFRAME_CONFIG  = {
    "root":"",
    "samples":"/audio"
}




NETWORK_CONFIG  = {
    "baseurl":"https://localhost:8080",
    "websocket":"ws://localhost:8080/entermedia/services/websocket/org/thoughtframe/websocket/BenchConnection?sessionid={tabID}",
    "apphome":"/thoughtframe"
}
